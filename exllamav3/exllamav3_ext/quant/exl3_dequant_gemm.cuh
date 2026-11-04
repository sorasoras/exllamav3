#pragma once

#include <ATen/Tensor.h>

// y = x @ decode(trellis), with the 4-bit weights dequantized tile-resident and never
// materialized in global memory. Same numerics as reconstruct() + hgemm_recon(), i.e. it
// consumes the input in the rotated basis produced by had_r_128(x, suh) and emits the
// rotated-basis output, which the caller rotates back with had_r_128(y, svh).
//
// A: [rows, size_k] half, the rotated input
// B: [k_blocks, n_tiles, 16*K + (half ? 8 : 0)] int16 packed trellis
// C: [rows, size_n] half or float, size_n must be a multiple of 128
//
// Returns false when the shape or codebook is not covered, so the caller can fall back to
// reconstruct + cuBLAS. No tensor cores required: this is a plain FFMA GEMM, which is what a
// pre-Volta device has to work with anyway.
bool exl3_dequant_gemm
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    float K,
    bool mcg,
    bool mul1
);
