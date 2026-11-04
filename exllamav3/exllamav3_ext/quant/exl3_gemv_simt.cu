#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "exl3_gemv_simt.cuh"
#include "exl3_dq.cuh"
#include "exl3_devctx.cuh"
#include "hadamard.cuh"
#include "bits_k.cuh"

// SIMT decode GEMV for pre-Volta devices.
//
// Layout: one warp owns one 16x16 trellis tile (16 decoded columns), one block owns eight of them.
// dq_dispatch<...>(tile, lane * 8, ...) hands lane l eight decoded values, and reconstruct_tile
// shows where they sit in the tile:
//
//   frag0[0] = (r0, colA) (r1, colA)      frag0[1] = (r2, colA) (r3, colA)
//   frag1[0] = (r0, colB) (r1, colB)      frag1[1] = (r2, colB) (r3, colB)
//
//   r0 = (l % 4) * 2, r1 = r0 + 1, r2 = r0 + 8, r3 = r0 + 9
//   colA = 2 * (l / 8) + (l & 4 ? 1 : 0), colB = colA + 8
//
// The four lanes that share (l / 8, l & 4) hold the same column in different rows, so the column
// sum is a two-step xor-shuffle reduction and no shared memory, ldmatrix or mma is needed.

namespace {

constexpr int SIMT_GEMV_WARPS = 8;
constexpr int SIMT_GEMV_THREADS = SIMT_GEMV_WARPS * 32;

// The tile walk is parallel over n only, which leaves too few warps in flight for a decode
// GEMV (n / 16 warps total, and a 1024-column tensor is 64 warps). The k dimension is split
// across blocks instead, each block writing its own slice of a float scratch buffer that a
// small second kernel reduces. Keeping the partial sums in separate slices rather than using
// atomics keeps the result deterministic.

template <int K, int cb, bool HALF>
__global__ __launch_bounds__(SIMT_GEMV_THREADS) void exl3_gemv_simt_kernel
(
    const half* __restrict__ A,
    const half* __restrict__ suh,
    const uint16_t* __restrict__ B,
    float* __restrict__ Y,
    int k_blocks,
    int n_tiles,
    int kb_per_split
)
{
    constexpr int packed_u16 = 16 * K + (HALF ? 8 : 0);
    constexpr int packed_u32 = packed_u16 / 2;

    const int lane = threadIdx.x & 31;
    const int nt = blockIdx.x * SIMT_GEMV_WARPS + (threadIdx.x >> 5);
    if (nt >= n_tiles) return;

    const int kb_begin = blockIdx.y * kb_per_split;
    const int kb_end = min(kb_begin + kb_per_split, k_blocks);
    if (kb_begin >= kb_end) return;

    const uint32_t* tile = (const uint32_t*)
        (B + ((size_t) kb_begin * n_tiles + nt) * packed_u16);

    const int r0 = (lane & 3) * 2;
    const int r1 = r0 + 1;
    const int r2 = r0 + 8;
    const int r3 = r0 + 9;
    const int colA = 2 * (lane >> 3) + ((lane & 4) ? 1 : 0);
    const int colB = colA + 8;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (int kb = kb_begin; kb < kb_end; ++kb)
    {
        FragB frag0, frag1;
        dq_dispatch<K, cb, HALF>(tile, lane * 8, frag0, frag1);

        const half* a = A + kb * 16;
        float a0 = __half2float(a[r0]);
        float a1 = __half2float(a[r1]);
        float a2 = __half2float(a[r2]);
        float a3 = __half2float(a[r3]);
        if (suh)
        {
            const half* s = suh + kb * 16;
            a0 *= __half2float(s[r0]);
            a1 *= __half2float(s[r1]);
            a2 *= __half2float(s[r2]);
            a3 *= __half2float(s[r3]);
        }

        float2 f00 = __half22float2(frag0[0]);
        float2 f01 = __half22float2(frag0[1]);
        float2 f10 = __half22float2(frag1[0]);
        float2 f11 = __half22float2(frag1[1]);

        acc0 = fmaf(a0, f00.x, acc0);
        acc0 = fmaf(a1, f00.y, acc0);
        acc0 = fmaf(a2, f01.x, acc0);
        acc0 = fmaf(a3, f01.y, acc0);

        acc1 = fmaf(a0, f10.x, acc1);
        acc1 = fmaf(a1, f10.y, acc1);
        acc1 = fmaf(a2, f11.x, acc1);
        acc1 = fmaf(a3, f11.y, acc1);

        tile += n_tiles * packed_u32;
    }

    acc0 += __shfl_xor_sync(0xffffffffu, acc0, 1);
    acc0 += __shfl_xor_sync(0xffffffffu, acc0, 2);
    acc1 += __shfl_xor_sync(0xffffffffu, acc1, 1);
    acc1 += __shfl_xor_sync(0xffffffffu, acc1, 2);

    if ((lane & 3) == 0)
    {
        const int split = blockIdx.y;
        Y[(size_t) split * n_tiles * 16 + nt * 16 + colA] = acc0;
        Y[(size_t) split * n_tiles * 16 + nt * 16 + colB] = acc1;
    }
}


// Reduce the per-split partial sums and apply the output scale vector.
__global__ void exl3_gemv_simt_finalize
(
    const float* __restrict__ Y,
    const half* __restrict__ svh,
    half* __restrict__ C,
    int n,
    int split
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = 0.0f;
    for (int s = 0; s < split; ++s) v += Y[(size_t) s * n + i];
    if (svh) v *= __half2float(svh[i]);
    C[i] = __float2half(v);
}

struct SimtArgs
{
    const half* A;
    const half* suh;
    const uint16_t* B;
    const half* svh;
    half* C;
    float* Y;
    int k_blocks;
    int n_tiles;
    int kb_per_split;
    int split;
    dim3 grid;
    dim3 fgrid;
    cudaStream_t stream;
};

template <int K, int cb>
void launch_simt(const SimtArgs& a)
{
    exl3_gemv_simt_kernel<K, cb, false>
        <<<a.grid, SIMT_GEMV_THREADS, 0, a.stream>>>
        (a.A, a.suh, a.B, a.Y, a.k_blocks, a.n_tiles, a.kb_per_split);
    exl3_gemv_simt_finalize
        <<<a.fgrid, 256, 0, a.stream>>>
        (a.Y, a.svh, a.C, a.n_tiles * 16, a.split);
}

#define SIMT_KIND(CBV, KV)                                               \
    case KV:                                                             \
        launch_simt<KV, CBV>(args);                                      \
        return true;

} // namespace

bool exl3_gemv_simt
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& A_had,
    const c10::optional<at::Tensor>& svh,
    int size_m,
    int size_k,
    int size_n,
    int K,
    bool half_k,
    int cb,
    bool c_fp32,
    cudaStream_t stream
)
{
    // One row, integer bitrate, fp16 out, no input Hadamard: everything else has a better path
    // (mul1 goes to the int8 dp4a GEMV, which is exact for its codebook) or no path at all.
    if (size_m != 1 || c_fp32 || half_k || A_had.has_value()) return false;
    if (K < 1 || K > 8 || (size_k % 16) != 0) return false;

    const c10::cuda::CUDAGuard device_guard(A.device());
    const int device = A.get_device();

    SimtArgs args;
    args.A = (const half*) A.data_ptr();
    args.B = (const uint16_t*) B.data_ptr();
    args.C = (half*) C.data_ptr();
    args.suh = suh.has_value() ? (const half*) suh->data_ptr() : nullptr;
    args.svh = svh.has_value() ? (const half*) svh->data_ptr() : nullptr;
    args.k_blocks = size_k / 16;
    args.n_tiles = size_n / 16;
    args.stream = stream;

    // Splitting k is the only way to fill the SMs here: the column decomposition yields
    // n / 16 warps in total, which is 64 for a 1024-column tensor on a 30-SM card.
    const int n_blocks = (args.n_tiles + SIMT_GEMV_WARPS - 1) / SIMT_GEMV_WARPS;
    const int num_sms = DevCtx::instance().get_num_sms(device);
    int split = (int) std::ceil((float) (4 * num_sms) / (float) n_blocks);
    split = std::max(1, std::min(split, std::max(1, args.k_blocks / 2)));
    args.split = split;
    args.kb_per_split = (args.k_blocks + split - 1) / split;
    args.grid = dim3(n_blocks, split);
    args.fgrid = dim3((size_n + 255) / 256);

    at::Tensor yf = at::empty({split, size_n}, A.options().dtype(at::kFloat));
    args.Y = (float*) yf.data_ptr();

    switch (cb)
    {
        case 0: switch (K)
        {
            SIMT_KIND(0, 1) SIMT_KIND(0, 2) SIMT_KIND(0, 3) SIMT_KIND(0, 4)
            SIMT_KIND(0, 5) SIMT_KIND(0, 6) SIMT_KIND(0, 7) SIMT_KIND(0, 8)
        } return false;
        case 1: switch (K)
        {
            SIMT_KIND(1, 1) SIMT_KIND(1, 2) SIMT_KIND(1, 3) SIMT_KIND(1, 4)
            SIMT_KIND(1, 5) SIMT_KIND(1, 6) SIMT_KIND(1, 7) SIMT_KIND(1, 8)
        } return false;
        case 2: switch (K)
        {
            SIMT_KIND(2, 1) SIMT_KIND(2, 2) SIMT_KIND(2, 3) SIMT_KIND(2, 4)
            SIMT_KIND(2, 5) SIMT_KIND(2, 6) SIMT_KIND(2, 7) SIMT_KIND(2, 8)
        } return false;
    }
    return false;
}

bool exl3_gemv_cb01
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& svh,
    const c10::optional<at::Tensor>& bias,
    float K,
    bool mcg,
    bool mul1
)
{
    if (!A.is_cuda() || A.dim() != 2 || A.size(0) != 1) return false;
    if (A.scalar_type() != at::kHalf || !A.is_contiguous()) return false;
    if (C.dim() != 2 || C.size(0) != 1 || !C.is_contiguous()) return false;
    if (C.scalar_type() != at::kHalf && C.scalar_type() != at::kFloat) return false;

    const int64_t size_k = A.size(1);
    const int64_t size_n = C.size(1);
    if ((size_k % 128) != 0 || (size_n % 128) != 0) return false;

    const BitsK bk = bits_from_K(K);
    if (bk.bits < 1 || bk.bits > 8) return false;

    const at::cuda::OptionalCUDAGuard device_guard(A.device());

    const c10::optional<at::Tensor> no_scale = c10::nullopt;
    at::Tensor xh = at::empty_like(A);
    had_r_128(A, xh, suh.has_value() ? suh : no_scale, no_scale, 1.0f);

    at::Tensor yh = at::empty({1, size_n}, A.options());
    const int cb = mul1 ? 2 : (mcg ? 1 : 0);
    if (!exl3_gemv_simt(xh, B, yh, no_scale, no_scale, no_scale,
                        (int) 1, (int) size_k, (int) size_n, bk.bits, bk.half, cb, false,
                        at::cuda::getCurrentCUDAStream().stream()))
        return false;

    // The output rotation is a real Hadamard transform, not a diagonal scale, so it has to run
    // as its own kernel; writing straight into a half output saves the extra copy.
    if (C.scalar_type() == at::kHalf)
        had_r_128(yh, C, no_scale, svh.has_value() ? svh : no_scale, 1.0f);
    else
    {
        had_r_128(yh, yh, no_scale, svh.has_value() ? svh : no_scale, 1.0f);
        C.copy_(yh);
    }

    if (bias.has_value() && bias->defined()) C.add_(*bias);
    return true;
}
