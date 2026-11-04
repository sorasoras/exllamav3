"""
Numerical validation for the pre-Volta (Pascal, sm_61) path.

Pascal has no tensor cores, so a single fatbin serves it by compiling the mma wrappers to
__trap() stubs and routing decode through the int8 dp4a GEMV, which is plain SIMT. Two things
must hold:

  * every tensor that can reach a kernel here has a working kernel (mul1, K <= 8, m == 1 for
    the K = 7/8 half-integer rates), and
  * everything else fails with a clear error instead of launching a kernel that traps and takes
    the process down.

The dp4a GEMV quantizes the activations to int8, so it is an approximation of the fp16 GEMM by
construction (that is true on Ampere too, where it ships on by default). These tests therefore
check the relative RMS error against reconstruct-then-matmul rather than exact equality: a
wrong fragment mapping or a broken scale decomposition shows up as tens of percent, not as a
fraction of a percent.

    python -m pytest tests/test_sm61_pascal.py -v
"""

import os

import pytest
import torch

# The GEMV is on by default for mul1, but make the test independent of the caller's environment
os.environ.setdefault("EXL3_INT8_GEMV", "2")

from exllamav3.ext import exllamav3_ext as ext

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

DEV = "cuda:0"
CC = torch.cuda.get_device_capability(DEV)
PRE_VOLTA = CC[0] < 7


def _make_trellis(k, n, K, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    trellis = torch.randint(
        0, 65536, (k // 16, n // 16, 16 * K), generator=g, dtype=torch.int32
    ).to(torch.int16).to(DEV)
    suh = (torch.randn(k, generator=g) * 0.1).to(torch.float16).to(DEV)
    svh = (torch.randn(n, generator=g) * 0.1).to(torch.float16).to(DEV)
    return trellis, suh, svh


def _reference(A, trellis, suh, svh, K, mcg, mul1):
    k = trellis.shape[0] * 16
    n = trellis.shape[1] * 16
    B = torch.empty((k, n), dtype=torch.float16, device=DEV)
    ext.reconstruct_had_slice(B, trellis, suh, svh, K, mcg, mul1, 0)
    return (A.float() @ B.float()).to(torch.float16)


def _rel_rms(a, b):
    err = (a.float() - b.float()).square().mean().sqrt()
    scale = b.float().square().mean().sqrt()
    return (err / scale).item()


@pytest.mark.parametrize("K", [1, 2, 3, 4, 5, 6, 7, 8])
def test_int8_gemv_matches_reconstruct(K):
    """The dp4a GEMV is the only quantized matmul that runs on Pascal; it must be right.

    K covers the staged-bit special cases (3, 5) as well as the plain ones, since the staging
    path is where the trellis decode and the dp4a fold have to agree. K = 7/8 only have
    cooperative-kernel instances, which is why decode is the single-row path.
    """
    k, n, m = 512, 512, 1
    trellis, suh, svh = _make_trellis(k, n, K, seed=K)
    A = (torch.randn((m, k), device=DEV) * 0.5).to(torch.float16)

    C = torch.empty((m, n), dtype=torch.float16, device=DEV)
    A_had = torch.empty_like(A)
    ext.exl3_gemm(A, trellis, C, suh, A_had, svh, 0, False, True, 0)
    torch.cuda.synchronize()

    assert torch.isfinite(C.float()).all(), f"K={K}: non-finite output"
    ref = _reference(A, trellis, suh, svh, K, False, True)
    rel = _rel_rms(C, ref)
    # int8 activation quantization alone costs a few tenths of a percent; anything in the
    # percent range means the weight side is wrong too.
    assert rel < 0.02, f"K={K}: relative RMS error {rel:.4f}"


@pytest.mark.parametrize("K", [2, 4, 6])
def test_int8_gemv_two_rows(K):
    """The shared-memory path covers m = 2 (batched verify); it must agree row by row."""
    k, n, m = 512, 512, 2
    trellis, suh, svh = _make_trellis(k, n, K, seed=200 + K)
    A = (torch.randn((m, k), device=DEV) * 0.5).to(torch.float16)
    C = torch.empty((m, n), dtype=torch.float16, device=DEV)
    A_had = torch.empty_like(A)
    ext.exl3_gemm(A, trellis, C, suh, A_had, svh, 0, False, True, 0)
    torch.cuda.synchronize()

    ref = _reference(A, trellis, suh, svh, K, False, True)
    rel = _rel_rms(C, ref)
    assert rel < 0.02, f"K={K}: relative RMS error {rel:.4f}"


@pytest.mark.parametrize("K", [1, 4, 8])
def test_int8_gemv_deterministic(K):
    """Repeat launches must be bit-identical: the staging path has no atomics or races."""
    k, n, m = 512, 512, 1
    trellis, suh, svh = _make_trellis(k, n, K, seed=100 + K)
    A = (torch.randn((m, k), device=DEV) * 0.5).to(torch.float16)
    A_had = torch.empty_like(A)

    outs = []
    for _ in range(4):
        C = torch.empty((m, n), dtype=torch.float16, device=DEV)
        ext.exl3_gemm(A, trellis, C, suh, A_had, svh, 0, False, True, 0)
        torch.cuda.synchronize()
        outs.append(C.clone())

    for i, o in enumerate(outs[1:], 1):
        assert torch.equal(outs[0], o), f"K={K}: launch {i} differs from launch 0"


@pytest.mark.skipif(not PRE_VOLTA, reason="asserts the pre-Volta guard")
def test_non_mul1_codebook_fails_cleanly():
    """cb0/cb1 weights have no kernel here; they must raise, not trap.

    A trap would kill the CUDA context and take every other device in the process with it, which
    on a mixed rig means losing the main card too.
    """
    k, n, K, m = 512, 512, 4, 1
    trellis, suh, svh = _make_trellis(k, n, K, seed=3)
    A = (torch.randn((m, k), device=DEV) * 0.5).to(torch.float16)
    C = torch.empty((m, n), dtype=torch.float16, device=DEV)
    A_had = torch.empty_like(A)

    with pytest.raises(RuntimeError, match="pre-Volta"):
        ext.exl3_gemm(A, trellis, C, suh, A_had, svh, 0, True, False, 0)
    # A raised TORCH_CHECK must leave the device usable: a trapped kernel would not
    torch.cuda.synchronize()
    assert (torch.ones(4, device=DEV) + 1).sum().item() == 8.0


@pytest.mark.skipif(not PRE_VOLTA, reason="asserts the pre-Volta guard")
def test_prefill_rows_fail_cleanly():
    """m > 8 has no tensor-core-free kernel; the Python layer routes those to reconstruct.

    Driving exl3_gemm directly with a prefill-sized m must still produce the actionable error.
    """
    k, n, K, m = 512, 512, 4, 16
    trellis, suh, svh = _make_trellis(k, n, K, seed=4)
    A = (torch.randn((m, k), device=DEV) * 0.5).to(torch.float16)
    C = torch.empty((m, n), dtype=torch.float16, device=DEV)
    A_had = torch.empty_like(A)

    with pytest.raises(RuntimeError, match="pre-Volta"):
        ext.exl3_gemm(A, trellis, C, suh, A_had, svh, 0, False, True, 0)


def test_reconstruct_slice_is_the_prefill_path():
    """Prefill on Pascal is reconstruct + cuBLAS, so the reconstruct kernels must run here."""
    k, n, K = 512, 512, 4
    trellis, suh, svh = _make_trellis(k, n, K, seed=5)

    B = torch.empty((k, n), dtype=torch.float16, device=DEV)
    ext.reconstruct_had_slice(B, trellis, suh, svh, K, False, True, 0)
    torch.cuda.synchronize()
    assert torch.isfinite(B.float()).all()
    assert B.abs().max().item() > 0

    # The non-fused slice path (one column block at a time) is what reconstruct_hgemm falls back
    # to for long prompts; it must run on this device too.
    B2 = torch.empty((k, n), dtype=torch.float16, device=DEV)
    ext.reconstruct_slice(B2, trellis, K, False, True, 0)
    torch.cuda.synchronize()
    assert torch.isfinite(B2.float()).all()
    assert B2.abs().max().item() > 0
