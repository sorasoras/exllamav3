#pragma once

#include <ATen/Tensor.h>
#include <cuda_runtime.h>

// Tensor-core-free GEMV for pre-Volta devices (sm_61).
//
// Both the regular GEMV and the block GEMM are mma kernels, and mma.sync/ldmatrix do not exist
// below sm_70. The int8 dp4a GEMV covers mul1 only, because its fold needs a codebook that is
// affine in the byte sum. cb0 (3INST) and cb1 (mcg) are not, so this path decodes the trellis
// with the same per-lane fragment layout reconstruct uses and takes the dot product with plain
// FFMA. It is a decode path: one row of A per launch, fp16 output, no Hadamard.
//
// Returns true if the operation was handled (false -> caller should fall through).

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
);

// One-call decode projection: Hadamard(suh) -> GEMV -> Hadamard(svh) -> bias.
//
// The Python path needs four torch calls plus two scratch allocations per projection, and a
// decode step performs ~200 projections, so the launch overhead ends up 3-4x the GPU time
// (measured: 8.4 ms of kernels against 34 ms wall per token). Doing the whole sequence in C++
// keeps one interpreter crossing per projection while issuing the same kernels.
//
// A is [1, size_k] fp16, B the trellis, C [1, size_n] fp16 or fp32. Returns false when the
// shape or dtype combination is not handled so the caller can fall back to reconstruct.
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
);
