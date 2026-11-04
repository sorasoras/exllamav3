// Standalone tuner for the int8 dp4a GEMM: 64x64 tile, 4 threads per staged row,
// register-prefetched next slab. BK selects the staging width (64 -> int4,
// 32 -> int2) and therefore the shared-memory footprint and resident blocks.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifndef BK
#define BK 64
#endif
#ifndef BM
#define BM 64
#endif
#ifndef BN
#define BN 64
#endif
#ifndef THREADS
#define THREADS 256
#endif
#ifndef TM
#define TM 4
#endif
#ifndef GROUP_N
#define GROUP_N 1
#endif
#ifndef TN
#define TN 4
#endif

#define PAD 4
#define STRIDE (BK + PAD)
#if BK == 64
typedef int4 StageT;
#define STAGE_INTS 4
#else
typedef int2 StageT;
#define STAGE_INTS 2
#endif
#define STRIDE_W (STRIDE / 4)

__global__ __launch_bounds__(THREADS)
void gemm_kernel(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                 void* __restrict__ Cout, const float* __restrict__ sa,
                 const float* __restrict__ sb, int size_m, int size_n, int size_k)
{
    __shared__ int32_t s_a[2][BM][STRIDE_W];
    __shared__ int32_t s_b[2][BN][STRIDE_W];

    const int tid = threadIdx.x;
    const int m_tiles = (size_m + BM - 1) / BM;
    const int n_tiles = (size_n + BN - 1) / BN;
    int mt, nt;
    if (GROUP_N <= 1) {
        mt = blockIdx.x / n_tiles;
        nt = blockIdx.x % n_tiles;
    } else {
        const int band = blockIdx.x / (GROUP_N * m_tiles);
        const int rem = blockIdx.x % (GROUP_N * m_tiles);
        nt = band * GROUP_N + rem / m_tiles;
        mt = rem % m_tiles;
        if (nt >= n_tiles) { nt = n_tiles - 1; }
    }
    const int m0 = mt * BM;
    const int n0 = nt * BN;
    const int k_blocks = size_k / BK;

    const int lrow = tid >> 2;
    const int lseg = (tid & 3) * STAGE_INTS;
    const int gm = m0 + lrow;
    const int gn = n0 + lrow;
    const bool ok_a = gm < size_m;
    const bool ok_b = gn < size_n;
    const int8_t* pa = A + (size_t) gm * size_k + lseg * 4;
    const int8_t* pb = B + (size_t) gn * size_k + lseg * 4;

    StageT ra, rb;

    auto fetch = [&](int kb) {
        const int off = kb * BK;
        ra = ok_a ? *(const StageT*) (pa + off) : StageT();
        rb = ok_b ? *(const StageT*) (pb + off) : StageT();
    };
    auto stash = [&](int buf) {
        if (ok_a) { s_a[buf][lrow][lseg+0]=((int*)&ra)[0]; s_a[buf][lrow][lseg+1]=((int*)&ra)[1];
                    #if STAGE_INTS == 4
                    s_a[buf][lrow][lseg+2]=((int*)&ra)[2]; s_a[buf][lrow][lseg+3]=((int*)&ra)[3];
                    #endif
                  }
        if (ok_b) { s_b[buf][lrow][lseg+0]=((int*)&rb)[0]; s_b[buf][lrow][lseg+1]=((int*)&rb)[1];
                    #if STAGE_INTS == 4
                    s_b[buf][lrow][lseg+2]=((int*)&rb)[2]; s_b[buf][lrow][lseg+3]=((int*)&rb)[3];
                    #endif
                  }
    };

    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warp_cols = BN / (TN * 4);
    const int wm = warp / warp_cols;
    const int wn = warp % warp_cols;
    const int r0 = wm * TM * 8 + (lane >> 2) * TM;
    const int c0 = wn * TN * 4 + (lane & 3) * TN;

    int acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0;

    fetch(0); stash(0); __syncthreads();

    for (int kb = 0; kb < k_blocks; ++kb)
    {
        const int buf = kb & 1;
        const bool more = (kb + 1) < k_blocks;
        if (more) fetch(kb + 1);

        #pragma unroll
        for (int kk = 0; kk < BK / 4; ++kk)
        {
            int av[TM], bv[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) av[i] = s_a[buf][r0 + i][kk];
            #pragma unroll
            for (int j = 0; j < TN; ++j) bv[j] = s_b[buf][c0 + j][kk];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = __dp4a(av[i], bv[j], acc[i][j]);
        }

        if (more) stash(buf ^ 1);
        __syncthreads();
    }

    const int gmo = m0 + r0, gno = n0 + c0;
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int mi = gmo + i;
        if (mi >= size_m) break;
        const float sx = sa[mi];
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int nj = gno + j;
            if (nj >= size_n) break;
            ((half*) Cout)[(size_t) mi * size_n + nj] = __float2half((float) acc[i][j] * sx * sb[nj]);
        }
    }
}

static float run(int m, int k, int n, const char* tag) {
    int8_t *A, *B; half* C; float *sa, *sb;
    cudaMalloc(&A, (size_t) m * k); cudaMalloc(&B, (size_t) n * k);
    cudaMalloc(&C, (size_t) m * n * 2);
    cudaMalloc(&sa, m * 4); cudaMalloc(&sb, n * 4);
    cudaMemset(A, 1, (size_t) m * k); cudaMemset(B, 1, (size_t) n * k);
    cudaMemset(sa, 0, m * 4); cudaMemset(sb, 0, n * 4);

    dim3 grid(((n + BN - 1) / BN) * ((m + BM - 1) / BM));
    for (int i = 0; i < 3; ++i) gemm_kernel<<<grid, THREADS>>>(A, B, C, sa, sb, m, n, k);
    cudaDeviceSynchronize();

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < 10; ++i) gemm_kernel<<<grid, THREADS>>>(A, B, C, sa, sb, m, n, k);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= 10;

    double tmac = (double) m * k * n / 1e12 / (ms / 1e3);
    printf("%-24s %4d x %5d x %5d  %8.2f ms  %5.2f TMAC/s (%2.0f%%)\n",
           tag, m, k, n, ms, tmac, tmac / 18.8 * 100);
    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(sa); cudaFree(sb);
    return ms;
}

int main() {
    char tag[128];
    snprintf(tag, sizeof(tag), "BM%d BN%d BK%d T%d %dx%d G%d", BM, BN, BK, THREADS, TM, TN, GROUP_N);
    double a = run(2045, 4096, 4096, tag);
    double b = run(2045, 4096, 14336, tag);
    double c = run(2045, 14336, 4096, tag);
    printf("%-24s TOTAL %.2f ms\n", tag, a + b + c);
    return 0;
}
