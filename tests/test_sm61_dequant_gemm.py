import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from exllamav3.ext import exllamav3_ext as ext


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] >= 7,
    reason="pre-Volta only",
)

CB_LIST = [(False, False), (True, False), (False, True)]  # (mcg, mul1)


def _make(k, n, K, dev="cuda:0", seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    trellis = torch.randint(0, 65536, (k // 16, n // 16, 16 * K), generator=g, dtype=torch.int32)
    trellis = trellis.to(torch.int16).to(dev)
    suh = (torch.rand(k, generator=g) * 0.5 + 0.75).to(torch.half).to(dev)
    svh = (torch.rand(n, generator=g) * 0.5 + 0.75).to(torch.half).to(dev)
    return trellis, suh, svh


def _reference(a, trellis, suh, svh, K, mcg, mul1):
    """reconstruct + cuBLAS, i.e. what LinearEXL3.reconstruct_hgemm does off the fused kernel."""
    k = trellis.shape[0] * 16
    n = trellis.shape[1] * 16
    w = torch.empty((k, n), dtype=torch.half, device="cuda")
    ext.reconstruct_slice(w, trellis, K, mcg, mul1, 0)
    xh = torch.empty_like(a)
    ext.had_r_128(a, xh, suh, None, 1.0)
    y = (xh.float() @ w.float()).to(torch.half)
    ext.had_r_128(y, y, None, svh, 1.0)
    return y


def _fused(a, trellis, suh, svh, K, mcg, mul1, out_dtype=torch.half):
    xh = torch.empty_like(a)
    ext.had_r_128(a, xh, suh, None, 1.0)
    c = torch.empty((a.shape[0], trellis.shape[1] * 16), dtype=out_dtype, device="cuda")
    ok = ext.exl3_dequant_gemm(xh, trellis, c, K, mcg, mul1)
    assert ok
    ext.had_r_128(c, c, None, svh, 1.0)
    return c


def _rel(c, ref):
    return ((c.float() - ref.float()).square().mean().sqrt()
            / ref.float().square().mean().sqrt()).item()


@pytest.mark.parametrize("K", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("mcg,mul1", CB_LIST)
def test_dequant_gemm_matches_reconstruct(K, mcg, mul1):
    k, n, m = 512, 512, 16
    trellis, suh, svh = _make(k, n, K, seed=K * 11 + (1 if mcg else 0) + (2 if mul1 else 0))
    a = torch.randn(m, k, device="cuda").half()

    y_ref = _reference(a, trellis, suh, svh, K, mcg, mul1)
    c = _fused(a, trellis, suh, svh, K, mcg, mul1)
    torch.cuda.synchronize()

    rel = _rel(c, y_ref)
    assert rel < 0.02, f"K={K} mcg={mcg} mul1={mul1}: rel rms {rel:.4f}"


@pytest.mark.parametrize("m", [1, 2, 15, 16, 31, 32, 33, 64, 477])
@pytest.mark.parametrize("k,n", [(1024, 1024), (2048, 1024), (1024, 2048), (3072, 3072)])
def test_dequant_gemm_shape_grid(m, k, n):
    K = 4
    trellis, suh, svh = _make(k, n, K, seed=m * 13 + k + n)
    a = torch.randn(m, k, device="cuda").half()

    y_ref = _reference(a, trellis, suh, svh, K, False, False)
    c = _fused(a, trellis, suh, svh, K, False, False)
    torch.cuda.synchronize()

    rel = _rel(c, y_ref)
    assert rel < 0.02, f"m={m} k={k} n={n}: rel rms {rel:.4f}"


def test_dequant_gemm_fp32_output():
    K, k, n, m = 4, 1024, 1024, 8
    trellis, suh, svh = _make(k, n, K, seed=77)
    a = torch.randn(m, k, device="cuda").half()

    y_ref = _reference(a, trellis, suh, svh, K, False, False)
    c = _fused(a, trellis, suh, svh, K, False, False, out_dtype=torch.float)
    torch.cuda.synchronize()

    assert c.dtype == torch.float
    assert _rel(c, y_ref) < 0.02


def test_dequant_gemm_mul1_half_rates():
    # 1.5 / 2.5 / 3.5 bpw: the trellis carries 8 extra uint16 per tile
    for K in (1.5, 2.5, 3.5):
        bits = int(K)
        k, n, m = 512, 512, 16
        g = torch.Generator(device="cpu").manual_seed(int(K * 10))
        trellis = torch.randint(0, 65536, (k // 16, n // 16, 16 * bits + 8), generator=g, dtype=torch.int32)
        trellis = trellis.to(torch.int16).to("cuda")
        suh = (torch.rand(k, generator=g) * 0.5 + 0.75).to(torch.half).to("cuda")
        svh = (torch.rand(n, generator=g) * 0.5 + 0.75).to(torch.half).to("cuda")
        a = torch.randn(m, k, device="cuda").half()

        y_ref = _reference(a, trellis, suh, svh, K, False, True)
        c = _fused(a, trellis, suh, svh, K, False, True)
        torch.cuda.synchronize()

        rel = _rel(c, y_ref)
        assert rel < 0.02, f"K={K}: rel rms {rel:.4f}"


def test_dequant_gemm_rejects_bad_shape():
    K, k, n = 4, 1024, 1024
    trellis, suh, svh = _make(k, n, K, seed=5)
    a = torch.randn(4, k, device="cuda").half()
    xh = torch.empty_like(a)
    ext.had_r_128(a, xh, suh, None, 1.0)
    c = torch.empty((4, 64), dtype=torch.half, device="cuda")
    assert not ext.exl3_dequant_gemm(xh, trellis, c, K, False, False)
