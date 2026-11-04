#pragma once

#include <ATen/ATen.h>

// Int8 (W8A8) GEMM for pre-Ampere prefill.
//
// The trellis codebooks cb0/cb1/cb2 are decoded once into an int8 weight matrix
// with one symmetric scale per output row, and activations are quantized per row
// at call time.  On GP102 dp4a retires 4 MACs per instruction at the FFMA issue
// rate (measured 18.8 vs 4.7 TMAC/s on the P40), so this is the only prefill path
// that can exceed the fp32 cuBLAS ceiling.  It is lossy, unlike the trellis
// kernels: the caller opts in.
//
// A: int8 [size_m, size_k] row-major, size_k % I8_BK == 0
// B: int8 [size_n, size_k] row-major (weights transposed once at build time)
// C: half or float [size_m, size_n]
// sa: float [size_m] (activation row scales), sb: float [size_n] (weight row scales)
// Returns false if the shapes are not covered, leaving C untouched.
bool exl3_int8_gemm(const at::Tensor& A,
                    const at::Tensor& B,
                    at::Tensor& C,
                    const at::Tensor& sa,
                    const at::Tensor& sb,
                    bool fp32);

// Per-row symmetric int8 quantization of a half activation matrix: Q = round(X / S)
// clamped to [-127, 127], S = absmax(row) / 127.  Fused into one kernel because the
// torch equivalent materializes several full-size temporaries.
bool exl3_quantize_rows_int8(const at::Tensor& X, at::Tensor& Q, at::Tensor& S);
