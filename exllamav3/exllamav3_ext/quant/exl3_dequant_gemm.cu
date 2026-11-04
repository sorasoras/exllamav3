#include <cuda_fp16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "exl3_dequant_gemm.cuh"
#include "exl3_dq.cuh"
#include "bits_k.cuh"
#include "exl3_devctx.cuh"

// Tensor-core-free GEMM with the EXL3 trellis decoded into shared memory.
//
// The fallback path on a device without tensor cores is reconstruct + cuBLAS: dequantize the
// whole weight matrix into an fp16 buffer (2 bytes per weight written) and then read it back
// (2 more), i.e. 4 bytes of traffic per weight. This kernel skips both passes by decoding a
// 16 x 128 slab of the trellis into smem and multiplying it there, so the weight traffic is the
// packed trellis itself (0.5 bytes per weight at 4 bpw).
//
// Tiling: 64 (m) x 128 (n) per block, k in slabs of 16. Eight warps: warp w owns output rows
// [8w, 8w+8) and decodes the 16-column tile w while lanes own 4 columns each, so one thread
// holds an 8 x 4 accumulator block and every k step costs one int4-equivalent input load and
// one float4 weight load for 32 FMAs. The 8 warps cover the 8 tiles that make up the 128-column
// block, so decode and MAC work need no separate producer stage.
//
// Under-occupied shapes (few blocks over n for a small row count) are split along k, with the
// partial sums reduced by a second kernel. That matters because a 1024-wide layer with 16 rows
// only produces 8 blocks on a 30-SM device, and the whole decode is latency bound there.
//
// Block order is (m_tile, n_tile) with the m tile varying fastest, so consecutive blocks over
// one n tile re-read the same packed slab, which stays resident in L2.

#define DQ_G_BM 64
#define DQ_G_BN 128
#define DQ_G_BK 16
#define DQ_G_THREADS 256
#define DQ_G_TM 8
#define DQ_G_TN 4
#define DQ_G_XPAD 8
#define DQ_G_WPAD 4
#define DQ_G_XSTRIDE (DQ_G_BM + DQ_G_XPAD)
#define DQ_G_WSTRIDE (DQ_G_BN + DQ_G_WPAD)
#define DQ_G_SPLIT_MAX 8
#define DQ_G_SPLIT_BYTES (256ull << 20)

template <int K, int cb, bool HALF>
__global__ __launch_bounds__(DQ_G_THREADS)
void exl3_dequant_gemm_kernel
(
    const half* __restrict__ A,
    const uint16_t* __restrict__ B,
    void* __restrict__ C,
    float* __restrict__ Y,
    int size_m,
    int size_k,
    int size_n,
    int n_tiles,
    int kb_per_split,
    bool fp32
)
{
    constexpr int packed_u16 = 16 * K + (HALF ? 8 : 0);
    constexpr int packed_u32 = packed_u16 / 2;
    constexpr int packed_i4 = packed_u16 / 8;

    // The input slab is staged as float: converting 8 halves per k step inside the inner loop
    // costs 8 cvt instructions for every 32 FMAs (~40% issue overhead), while the staging pass
    // touches each element once.
    __shared__ float s_x[DQ_G_BK][DQ_G_XSTRIDE];
    __shared__ float s_w[DQ_G_BK][DQ_G_WSTRIDE];
    __shared__ uint32_t s_packed[8][packed_u32];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;                 // 8 warps: 8 output rows and 16 columns each

    const int m0 = blockIdx.x * DQ_G_BM;
    const int n0 = blockIdx.y * DQ_G_BN;
    const int nt0 = n0 >> 4;                   // first 16-column tile of this block
    const int z = blockIdx.z;
    const int split = gridDim.z;

    float acc[DQ_G_TM][DQ_G_TN];
    #pragma unroll
    for (int i = 0; i < DQ_G_TM; ++i)
        #pragma unroll
        for (int j = 0; j < DQ_G_TN; ++j)
            acc[i][j] = 0.0f;

    const int k_blocks = size_k >> 4;
    const int kb_begin = z * kb_per_split;
    const int kb_end = min(kb_begin + kb_per_split, k_blocks);

    for (int kb = kb_begin; kb < kb_end; ++kb)
    {
        // stage the input slab, transposed to [k][m] so the inner loop reads 8 contiguous
        // halves per thread; 128 threads cover 64 rows x 16 k as 8 halves each
        if (tid < DQ_G_BM * 2)
        {
            const int r = tid >> 1;
            const int kv = (tid & 1) << 3;
            const int gr = m0 + r;
            if (gr < size_m)
            {
                const half* src = A + (size_t) gr * size_k + kb * 16 + kv;
                const half2 h0 = *(const half2*) (src + 0);
                const half2 h1 = *(const half2*) (src + 2);
                const half2 h2 = *(const half2*) (src + 4);
                const half2 h3 = *(const half2*) (src + 6);
                const float2 f0 = __half22float2(h0);
                const float2 f1 = __half22float2(h1);
                const float2 f2 = __half22float2(h2);
                const float2 f3 = __half22float2(h3);
                s_x[kv + 0][r] = f0.x;
                s_x[kv + 1][r] = f0.y;
                s_x[kv + 2][r] = f1.x;
                s_x[kv + 3][r] = f1.y;
                s_x[kv + 4][r] = f2.x;
                s_x[kv + 5][r] = f2.y;
                s_x[kv + 6][r] = f3.x;
                s_x[kv + 7][r] = f3.y;
            }
            else
            {
                #pragma unroll
                for (int j = 0; j < 8; ++j) s_x[kv + j][r] = 0.0f;
            }
        }

        // stage the weight slab: each warp decodes its own 16 k x 16 n tile
        {
            const uint16_t* g = B + ((size_t) kb * n_tiles + nt0 + warp) * packed_u16;
            for (int t = lane; t < packed_i4; t += 32)
                ((int4*) s_packed[warp])[t] = ((const int4*) g)[t];
            __syncwarp();

            FragB f0, f1;
            dq_dispatch<K, cb, HALF>(s_packed[warp], lane * 8, f0, f1);

            const int r0 = (lane & 3) * 2;
            const int r1 = r0 + 1;
            const int r2 = r0 + 8;
            const int r3 = r0 + 9;
            const int cA = warp * 16 + 2 * (lane >> 3) + ((lane & 4) ? 1 : 0);
            const int cB = cA + 8;

            s_w[r0][cA] = __low2float(f0[0]);
            s_w[r1][cA] = __high2float(f0[0]);
            s_w[r2][cA] = __low2float(f0[1]);
            s_w[r3][cA] = __high2float(f0[1]);
            s_w[r0][cB] = __low2float(f1[0]);
            s_w[r1][cB] = __high2float(f1[0]);
            s_w[r2][cB] = __low2float(f1[1]);
            s_w[r3][cB] = __high2float(f1[1]);
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < DQ_G_BK; ++kk)
        {
            const float* xr = s_x[kk] + warp * DQ_G_TM;
            const float* wr = s_w[kk] + lane * DQ_G_TN;

            const float4 xa = *(const float4*) (xr + 0);
            const float4 xb = *(const float4*) (xr + 4);
            const float4 wv = *(const float4*) wr;

            acc[0][0] = fmaf(xa.x, wv.x, acc[0][0]);
            acc[0][1] = fmaf(xa.x, wv.y, acc[0][1]);
            acc[0][2] = fmaf(xa.x, wv.z, acc[0][2]);
            acc[0][3] = fmaf(xa.x, wv.w, acc[0][3]);
            acc[1][0] = fmaf(xa.y, wv.x, acc[1][0]);
            acc[1][1] = fmaf(xa.y, wv.y, acc[1][1]);
            acc[1][2] = fmaf(xa.y, wv.z, acc[1][2]);
            acc[1][3] = fmaf(xa.y, wv.w, acc[1][3]);
            acc[2][0] = fmaf(xa.z, wv.x, acc[2][0]);
            acc[2][1] = fmaf(xa.z, wv.y, acc[2][1]);
            acc[2][2] = fmaf(xa.z, wv.z, acc[2][2]);
            acc[2][3] = fmaf(xa.z, wv.w, acc[2][3]);
            acc[3][0] = fmaf(xa.w, wv.x, acc[3][0]);
            acc[3][1] = fmaf(xa.w, wv.y, acc[3][1]);
            acc[3][2] = fmaf(xa.w, wv.z, acc[3][2]);
            acc[3][3] = fmaf(xa.w, wv.w, acc[3][3]);
            acc[4][0] = fmaf(xb.x, wv.x, acc[4][0]);
            acc[4][1] = fmaf(xb.x, wv.y, acc[4][1]);
            acc[4][2] = fmaf(xb.x, wv.z, acc[4][2]);
            acc[4][3] = fmaf(xb.x, wv.w, acc[4][3]);
            acc[5][0] = fmaf(xb.y, wv.x, acc[5][0]);
            acc[5][1] = fmaf(xb.y, wv.y, acc[5][1]);
            acc[5][2] = fmaf(xb.y, wv.z, acc[5][2]);
            acc[5][3] = fmaf(xb.y, wv.w, acc[5][3]);
            acc[6][0] = fmaf(xb.z, wv.x, acc[6][0]);
            acc[6][1] = fmaf(xb.z, wv.y, acc[6][1]);
            acc[6][2] = fmaf(xb.z, wv.z, acc[6][2]);
            acc[6][3] = fmaf(xb.z, wv.w, acc[6][3]);
            acc[7][0] = fmaf(xb.w, wv.x, acc[7][0]);
            acc[7][1] = fmaf(xb.w, wv.y, acc[7][1]);
            acc[7][2] = fmaf(xb.w, wv.z, acc[7][2]);
            acc[7][3] = fmaf(xb.w, wv.w, acc[7][3]);
        }

        __syncthreads();
    }

    // ---- epilogue: direct write when the k range was not split, else float partials
    #pragma unroll
    for (int i = 0; i < DQ_G_TM; ++i)
    {
        const int gr = m0 + warp * DQ_G_TM + i;
        if (gr >= size_m) continue;
        const int gc0 = n0 + lane * DQ_G_TN;
        #pragma unroll
        for (int j = 0; j < DQ_G_TN; ++j)
        {
            const int gc = gc0 + j;
            if (gc >= size_n) continue;
            if (Y)
            {
                Y[((size_t) z * size_m + gr) * size_n + gc] = acc[i][j];
            }
            else
            {
                const size_t idx = (size_t) gr * size_n + gc;
                if (fp32) ((float*) C)[idx] = acc[i][j];
                else ((half*) C)[idx] = __float2half(acc[i][j]);
            }
        }
    }
}

__global__ void exl3_dequant_gemm_finalize
(
    const float* __restrict__ Y,
    void* __restrict__ C,
    int total,
    int split,
    bool fp32
)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float v = 0.0f;
    for (int s = 0; s < split; ++s) v += Y[(size_t) s * total + i];
    if (fp32) ((float*) C)[i] = v;
    else ((half*) C)[i] = __float2half(v);
}

// Instance table, same indexing as reconstruct.cu: cb * 8 + K - 1 for integer bitrates, and
// 24 + K - 1 for the half-integer rates 1.5 / 2.5 / 3.5 (mul1 only).
#define __(i, cb) exl3_dequant_gemm_kernel<i, cb, false>
constexpr auto dq_gemm_instances = std::array
{
    __(1, 0), __(2, 0), __(3, 0), __(4, 0), __(5, 0), __(6, 0), __(7, 0), __(8, 0),
    __(1, 1), __(2, 1), __(3, 1), __(4, 1), __(5, 1), __(6, 1), __(7, 1), __(8, 1),
    __(1, 2), __(2, 2), __(3, 2), __(4, 2), __(5, 2), __(6, 2), __(7, 2), __(8, 2),
    exl3_dequant_gemm_kernel<1, 2, true>,
    exl3_dequant_gemm_kernel<2, 2, true>,
    exl3_dequant_gemm_kernel<3, 2, true>
};
#undef __

bool exl3_dequant_gemm
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    float K_,
    bool mcg,
    bool mul1
)
{
    if (!A.is_cuda() || !B.is_cuda() || !C.is_cuda()) return false;
    if (A.dim() != 2 || B.dim() != 3 || C.dim() != 2) return false;
    if (A.scalar_type() != at::kHalf) return false;
    if (C.scalar_type() != at::kHalf && C.scalar_type() != at::kFloat) return false;
    if (!A.is_contiguous() || !B.is_contiguous() || !C.is_contiguous()) return false;

    const BitsK bk = bits_from_K(K_);
    if (bk.half && !mul1) return false;

    const int size_m = (int) A.size(0);
    const int size_k = (int) A.size(1);
    const int size_n = (int) C.size(1);
    if (size_m < 1 || size_k < 1 || size_n < 1) return false;
    if ((size_k & 15) || (size_n & 127)) return false;
    if ((int) C.size(0) != size_m) return false;
    if ((int) B.size(0) * 16 != size_k || (int) B.size(1) * 16 != size_n) return false;
    if ((int) B.size(2) != 16 * bk.bits + (bk.half ? 8 : 0)) return false;

    int cbi = bk.bits - 1;
    if (bk.half) cbi += 24;
    else if (mcg) cbi += 8;
    else if (mul1) cbi += 16;
    if (cbi < 0 || cbi >= (int) dq_gemm_instances.size()) return false;

    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    const int device = A.get_device();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const int m_tiles = (size_m + DQ_G_BM - 1) / DQ_G_BM;
    const int n_tiles_tot = size_n / DQ_G_BN;
    const int k_blocks = size_k / 16;
    const int num_sms = DevCtx::instance().get_num_sms(device);

    // split k when the m x n grid alone cannot fill the device
    int split = 1;
    const int blocks = m_tiles * n_tiles_tot;
    if (blocks < num_sms)
    {
        split = std::min(DQ_G_SPLIT_MAX, (num_sms + blocks - 1) / blocks);
        split = std::min(split, k_blocks);
        if ((size_t) split * size_m * size_n * sizeof(float) > DQ_G_SPLIT_BYTES) split = 1;
    }
    const int kb_per_split = (k_blocks + split - 1) / split;

    at::Tensor ys;
    float* y_ptr = nullptr;
    if (split > 1)
    {
        ys = at::empty({split, size_m, size_n}, A.options().dtype(at::kFloat));
        y_ptr = (float*) ys.data_ptr();
    }

    const dim3 block(DQ_G_THREADS);
    const dim3 grid(m_tiles, n_tiles_tot, split);

    dq_gemm_instances[cbi]<<<grid, block, 0, stream>>>
    (
        (const half*) A.data_ptr(),
        (const uint16_t*) B.data_ptr(),
        C.data_ptr(),
        y_ptr,
        size_m,
        size_k,
        size_n,
        (int) B.size(1),
        kb_per_split,
        C.scalar_type() == at::kFloat
    );
    cuda_check(cudaPeekAtLastError());

    if (split > 1)
    {
        const int total = size_m * size_n;
        const int threads = 256;
        exl3_dequant_gemm_finalize<<<(total + threads - 1) / threads, threads, 0, stream>>>
        (
            y_ptr,
            C.data_ptr(),
            total,
            split,
            C.scalar_type() == at::kFloat
        );
        cuda_check(cudaPeekAtLastError());
    }
    return true;
}
