#include <cuda_fp16.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "exl3_int8_gemm.cuh"

#define I8_BM 64
#define I8_BN 64
#define I8_BK 64
#define I8_THREADS 256
#define I8_TM 4
#define I8_TN 4
#define I8_GROUP_N 8

// Row stride in bytes.  68 = 17 words, and the k-loop reads 4 rows per thread
// spaced 4 apart, so the warp's 8 thread-rows land on banks (4 * 17 * ty) % 32
// = 4 * ty, i.e. 8 distinct banks.  A stride that is a multiple of 16 would be
// nicer for the staging stores but makes 4*s even and collapses those eight
// rows onto two banks, so the staging writes are split into int stores instead.
#define I8_PAD 4
#define I8_STRIDE (I8_BK + I8_PAD)
#define I8_STRIDE_W (I8_STRIDE / 4)

__global__ __launch_bounds__(I8_THREADS)
void exl3_int8_gemm_kernel(const int8_t* __restrict__ A,
                           const int8_t* __restrict__ B,
                           void* __restrict__ Cout,
                           const float* __restrict__ sa,
                           const float* __restrict__ sb,
                           int size_m, int size_n, int size_k, bool fp32)
{
    __shared__ int32_t s_a[2][I8_BM][I8_STRIDE_W];
    __shared__ int32_t s_b[2][I8_BN][I8_STRIDE_W];

    const int tid = threadIdx.x;

    // Blocks are dispatched in an n-band order: within a band of I8_GROUP_N
    // output tiles the m index varies fastest, so those tiles' rows stay
    // resident in L2 while every m tile streams past.  Launching n-fastest
    // instead re-reads the whole n side once per m tile, which is what made the
    // wide-k shapes (14336 -> 4096) run at 47% of the dp4a peak instead of 57%.
    const int m_tiles = (size_m + I8_BM - 1) / I8_BM;
    const int n_tiles = (size_n + I8_BN - 1) / I8_BN;
    const int band = blockIdx.x / (I8_GROUP_N * m_tiles);
    const int rem = blockIdx.x % (I8_GROUP_N * m_tiles);
    const int nt = band * I8_GROUP_N + rem / m_tiles;
    if (nt >= n_tiles) return;
    const int m0 = (rem % m_tiles) * I8_BM;
    const int n0 = nt * I8_BN;
    const int k_blocks = size_k / I8_BK;

    // 4 threads per row, 16 bytes each. The global fetch and the shared-memory
    // store are separate steps so the next slab's loads are in flight while this
    // slab is being multiplied: with them fused the thread stalls on the load
    // before it can issue any dp4a, which cost us ~45% of the peak rate.
    const int lrow = tid >> 2;
    const int lseg = (tid & 3) * 4;      // in int units
    const int gm = m0 + lrow;
    const int gn = n0 + lrow;
    const bool ok_a = gm < size_m;
    const bool ok_b = gn < size_n;
    const int8_t* pa = A + (size_t) gm * size_k + lseg * 4;
    const int8_t* pb = B + (size_t) gn * size_k + lseg * 4;

    int4 ra = make_int4(0, 0, 0, 0);
    int4 rb = make_int4(0, 0, 0, 0);

    auto fetch = [&](int kb)
    {
        const int k0 = kb * I8_BK;
        ra = ok_a ? *(const int4*) (pa + k0) : make_int4(0, 0, 0, 0);
        rb = ok_b ? *(const int4*) (pb + k0) : make_int4(0, 0, 0, 0);
    };

    auto stash = [&](int buf)
    {
        s_a[buf][lrow][lseg + 0] = ra.x;
        s_a[buf][lrow][lseg + 1] = ra.y;
        s_a[buf][lrow][lseg + 2] = ra.z;
        s_a[buf][lrow][lseg + 3] = ra.w;
        s_b[buf][lrow][lseg + 0] = rb.x;
        s_b[buf][lrow][lseg + 1] = rb.y;
        s_b[buf][lrow][lseg + 2] = rb.z;
        s_b[buf][lrow][lseg + 3] = rb.w;
    };

    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm = warp >> 2;
    const int wn = warp & 3;
    const int r0 = wm * 32 + (lane >> 2) * I8_TM;
    const int c0 = wn * 16 + (lane & 3) * I8_TN;

    int acc[I8_TM][I8_TN];
    #pragma unroll
    for (int i = 0; i < I8_TM; ++i)
        #pragma unroll
        for (int j = 0; j < I8_TN; ++j) acc[i][j] = 0;

    fetch(0);
    stash(0);
    __syncthreads();

    for (int kb = 0; kb < k_blocks; ++kb)
    {
        const int buf = kb & 1;
        const bool more = (kb + 1) < k_blocks;
        if (more) fetch(kb + 1);

        #pragma unroll
        for (int kk = 0; kk < I8_BK / 4; ++kk)
        {
            int av[I8_TM], bv[I8_TN];
            #pragma unroll
            for (int i = 0; i < I8_TM; ++i)
                av[i] = s_a[buf][r0 + i][kk];
            #pragma unroll
            for (int j = 0; j < I8_TN; ++j)
                bv[j] = s_b[buf][c0 + j][kk];
            #pragma unroll
            for (int i = 0; i < I8_TM; ++i)
                #pragma unroll
                for (int j = 0; j < I8_TN; ++j)
                    acc[i][j] = __dp4a(av[i], bv[j], acc[i][j]);
        }

        if (more) stash(buf ^ 1);
        __syncthreads();
    }

    const int gm_out = m0 + r0;
    const int gn_out = n0 + c0;
    #pragma unroll
    for (int i = 0; i < I8_TM; ++i)
    {
        const int mi = gm_out + i;
        if (mi >= size_m) break;
        const float sx = sa[mi];
        #pragma unroll
        for (int j = 0; j < I8_TN; ++j)
        {
            const int nj = gn_out + j;
            if (nj >= size_n) break;
            const float v = (float) acc[i][j] * sx * sb[nj];
            const size_t o = (size_t) mi * size_n + nj;
            if (fp32) ((float*) Cout)[o] = v;
            else ((half*) Cout)[o] = __float2half(v);
        }
    }
}

// Per-row activation quantization: one block per row, absmax reduction then a
// single quantizing pass. Two global reads of the row, no temporaries; the torch
// formulation costs ~1.2 ms for a 2045x4096 activation against ~0.1 here.
__global__ void exl3_quantize_rows_int8_kernel(const half* __restrict__ X,
                                              int8_t* __restrict__ Q,
                                              float* __restrict__ S,
                                              int rows, int cols)
{
    const int row = blockIdx.x;
    if (row >= rows) return;

    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const half* xr = X + (size_t) row * cols;

    float amax = 0.0f;
    for (int i = tid; i < cols; i += nt)
        amax = fmaxf(amax, fabsf(__half2float(xr[i])));

    __shared__ float s_red[I8_THREADS / 32];
    const int warp = tid >> 5;
    const int lane = tid & 31;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if (lane == 0) s_red[warp] = amax;
    __syncthreads();

    if (warp == 0)
    {
        amax = (lane < (nt / 32)) ? s_red[lane] : 0.0f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        if (lane == 0) s_red[0] = amax;
    }
    __syncthreads();

    const float scale = fmaxf(s_red[0], 1e-30f) / 127.0f;
    const float inv = 1.0f / scale;
    if (tid == 0) S[row] = scale;

    int8_t* qr = Q + (size_t) row * cols;
    for (int i = tid; i < cols; i += nt)
    {
        int q = __float2int_rn(__half2float(xr[i]) * inv);
        q = q > 127 ? 127 : (q < -127 ? -127 : q);
        qr[i] = (int8_t) q;
    }
}

bool exl3_int8_gemm(const at::Tensor& A,
                    const at::Tensor& B,
                    at::Tensor& C,
                    const at::Tensor& sa,
                    const at::Tensor& sb,
                    bool fp32)
{
    if (!A.is_cuda() || A.dim() != 2 || B.dim() != 2 || C.dim() != 2) return false;
    if (!A.is_contiguous() || !B.is_contiguous() || !C.is_contiguous()) return false;
    if (A.scalar_type() != at::kChar || B.scalar_type() != at::kChar) return false;
    if (sa.scalar_type() != at::kFloat || sb.scalar_type() != at::kFloat) return false;
    if (fp32 ? (C.scalar_type() != at::kFloat) : (C.scalar_type() != at::kHalf)) return false;

    const int64_t size_m = A.size(0);
    const int64_t size_k = A.size(1);
    const int64_t size_n = B.size(0);
    if (B.size(1) != size_k || C.size(0) != size_m || C.size(1) != size_n) return false;
    if (sa.numel() != size_m || sb.numel() != size_n) return false;
    if (size_m <= 0 || size_n <= 0 || (size_k % I8_BK) != 0) return false;

    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const int m_tiles = (int) ((size_m + I8_BM - 1) / I8_BM);
    const int n_tiles = (int) ((size_n + I8_BN - 1) / I8_BN);
    exl3_int8_gemm_kernel<<<(unsigned) (m_tiles * n_tiles), I8_THREADS, 0, stream>>>(
        (const int8_t*) A.data_ptr(),
        (const int8_t*) B.data_ptr(),
        C.data_ptr(),
        sa.data_ptr<float>(),
        sb.data_ptr<float>(),
        (int) size_m, (int) size_n, (int) size_k, fp32);
    TORCH_CHECK(cudaPeekAtLastError() == cudaSuccess, "exl3_int8_gemm: launch failed");
    return true;
}

bool exl3_quantize_rows_int8(const at::Tensor& X, at::Tensor& Q, at::Tensor& S)
{
    if (!X.is_cuda() || X.dim() != 2 || X.scalar_type() != at::kHalf) return false;
    if (!X.is_contiguous() || !Q.is_contiguous()) return false;
    if (Q.scalar_type() != at::kChar || S.scalar_type() != at::kFloat) return false;
    if (Q.sizes() != X.sizes() || S.numel() != X.size(0)) return false;
    if (X.size(1) <= 0) return false;

    const int rows = (int) X.size(0);
    const int cols = (int) X.size(1);

    const at::cuda::OptionalCUDAGuard device_guard(X.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    exl3_quantize_rows_int8_kernel<<<rows, I8_THREADS, 0, stream>>>(
        (const half*) X.data_ptr(), (int8_t*) Q.data_ptr(), S.data_ptr<float>(), rows, cols);
    TORCH_CHECK(cudaPeekAtLastError() == cudaSuccess, "exl3_quantize_rows_int8: launch failed");
    return true;
}
