#pragma once

// Architecture feature gates.
//
// EXL3 was written against sm_80+. Turing (sm_75) is missing three things the EXL3 GEMM
// relies on, and each one is handled in a different place:
//
//   1. cp.async         -> ptx.cuh falls back to plain synchronous 16 B smem stores. The
//                          pipeline's existing __syncthreads() calls already provide the
//                          ordering the async waits used to provide, so the fallback is
//                          correct, just without global->shared overlap.
//   2. mma.m16n8k16     -> ptx.cuh issues two mma.m16n8k8 instead. The k=16 fragment splits
//                          exactly into the two k=8 fragments (A: {a0,a1} then {a2,a3},
//                          B: b0 then b1), so this is bit-equivalent, not an approximation.
//   3. 90 KB shared mem -> Turing caps dynamic shared memory at 64 KB per block. The limit is
//                          a host-side launch parameter, so it is resolved per device at
//                          runtime (DevCtx::get_smem_max) and shapes that do not fit are
//                          filtered out of kernel selection rather than failing at launch.
//
// ldmatrix, __nanosleep, __dp4a and tanh.approx.f32 are all sm_75-capable and need no gate.
//
// Pascal (sm_61) goes one step further: it has no tensor cores at all, so every mma, ldmatrix,
// scoped acquire/release and __nanosleep below sm_70 is out. The tensor-core entry points are
// kept in the same fatbin but compiled to __trap() stubs; the dispatcher is responsible for
// never selecting them. Decode runs through the int8 dp4a GEMV (exl3_gemv_int8) and prefill
// through reconstruct + cuBLAS, both of which are plain SIMT and need nothing new here.
//
// EXL3_SM75 keeps its existing meaning ("pre-Ampere"), so the cp.async fallbacks already
// cover Pascal; EXL3_SM61 only gates the Volta+ instructions.

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
    #define EXL3_SM61 1
#else
    #define EXL3_SM61 0
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
    #define EXL3_SM75 1
#else
    #define EXL3_SM75 0
#endif

// __nanosleep is sm_70+. Below that it is not declared at all, so callers go through this
// macro; the spin loops that use it re-read a flag every iteration and are correct without it.
#if EXL3_SM61
    #define EXL3_NANOSLEEP(ns) ((void) (ns))
#else
    #define EXL3_NANOSLEEP(ns) __nanosleep(ns)
#endif

// __grid_constant__ is rejected by ptxas below compute_70. Without it the parameter is simply
// copied to local memory, which is what pre-Volta did anyway, so dropping the annotation keeps
// the kernel correct there (only the const-address guarantee is lost).
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
    #define EXL3_GRID_CONSTANT
#else
    #define EXL3_GRID_CONSTANT __grid_constant__
#endif

// Dynamic shared memory per block. SMEM_MAX stays at the sm_86 value because it only bounds
// the compile-time static_assert in exl3_gemm_inner.cuh, which must keep accepting every
// template instantiation. What actually gets requested at launch is the per-device value.
#define EXL3_SMEM_MAX_DEFAULT (90 * 1024)
#define EXL3_SMEM_MAX_SM75    (64 * 1024)
