// Peak MAC throughput on this GPU: dp4a vs FFMA, register-resident so the
// answer is the instruction rate, not memory.
#include <cstdio>
#include <cuda_runtime.h>

#define NACC 8

__global__ void dp4a_bench(int iters, int* out) {
    int a[NACC], b[NACC], c[NACC];
    #pragma unroll
    for (int i = 0; i < NACC; ++i) {
        a[i] = threadIdx.x * 7 + i * 3 + 1;
        b[i] = blockIdx.x * 11 + i * 5 + 2;
        c[i] = 0;
    }
    for (int t = 0; t < iters; ++t) {
        #pragma unroll
        for (int i = 0; i < NACC; ++i) c[i] = __dp4a(a[i], b[i], c[i]);
    }
    int s = 0;
    #pragma unroll
    for (int i = 0; i < NACC; ++i) s += c[i];
    if (s == 0x7fffffff) out[0] = s;
}

__global__ void fma_bench(int iters, int* out) {
    float a[NACC], b[NACC], c[NACC];
    #pragma unroll
    for (int i = 0; i < NACC; ++i) {
        a[i] = threadIdx.x * 0.5f + i;
        b[i] = blockIdx.x * 0.25f + i * 2.0f;
        c[i] = 0.0f;
    }
    for (int t = 0; t < iters; ++t) {
        #pragma unroll
        for (int i = 0; i < NACC; ++i) c[i] = fmaf(a[i], b[i], c[i]);
    }
    float s = 0;
    #pragma unroll
    for (int i = 0; i < NACC; ++i) s += c[i];
    if (s == 1.0f) out[0] = 1;
}

int main() {
    int dev = 0;
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);
    printf("%s  SMs=%d  clock=%.0f MHz\n", p.name, p.multiProcessorCount, p.clockRate / 1000.0);

    const int threads = 256;
    const int blocks = p.multiProcessorCount * 8;
    const int iters = 200000;
    int* d_out; int h_out = 0;
    cudaMalloc(&d_out, sizeof(int));

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);

    for (int rep = 0; rep < 2; ++rep) {
        cudaEventRecord(e0);
        dp4a_bench<<<blocks, threads>>>(iters, d_out);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms_d; cudaEventElapsedTime(&ms_d, e0, e1);

        cudaEventRecord(e0);
        fma_bench<<<blocks, threads>>>(iters, d_out);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms_f; cudaEventElapsedTime(&ms_f, e0, e1);

        double dp4a_ops = (double)blocks * threads * iters * NACC;
        printf("dp4a: %.2f ms  %.2f T instr/s  %.2f TMAC/s\n",
               ms_d, dp4a_ops / (ms_d * 1e9), dp4a_ops * 4.0 / (ms_d * 1e9));
        printf("fma : %.2f ms  %.2f T instr/s  %.2f TMAC/s (%.2f TFLOP/s)\n",
               ms_f, dp4a_ops / (ms_f * 1e9), dp4a_ops / (ms_f * 1e9),
               dp4a_ops * 2.0 / (ms_f * 1e9));
        printf("MAC ratio dp4a/fma = %.2fx\n\n", (dp4a_ops * 4.0 / ms_d) / (dp4a_ops / ms_f));
    }
    return 0;
}
