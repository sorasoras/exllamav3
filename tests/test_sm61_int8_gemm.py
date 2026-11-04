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


def _weights_int8(trellis, K, mcg, mul1):
    """What LinearEXL3._int8_weights does: stored-basis weights, per-output-row scale."""
    k = trellis.shape[0] * 16
    n = trellis.shape[1] * 16
    w = torch.empty((k, n), dtype=torch.half, device="cuda")
    ext.reconstruct_slice(w, trellis, K, mcg, mul1, 0)
    wf = w.float()
    amax = wf.abs().amax(dim=0)
    s = (amax / 127.0).clamp_min(1e-8)
    q = torch.round(wf / s).clamp_(-127, 127).to(torch.int8).t().contiguous()
    return q, s


def _reference(a, trellis, suh, svh, K, mcg, mul1):
    k = trellis.shape[0] * 16
    n = trellis.shape[1] * 16
    w = torch.empty((k, n), dtype=torch.half, device="cuda")
    ext.reconstruct_slice(w, trellis, K, mcg, mul1, 0)
    xh = torch.empty_like(a)
    ext.had_r_128(a, xh, suh, None, 1.0)
    y = (xh.float() @ w.float()).to(torch.half)
    ext.had_r_128(y, y, None, svh, 1.0)
    return y


def _int8_gemm_path(a, trellis, suh, svh, K, mcg, mul1):
    q, s = _weights_int8(trellis, K, mcg, mul1)
    xh = torch.empty_like(a)
    ext.had_r_128(a, xh, suh, None, 1.0)
    amax = xh.abs().amax(dim=1)
    sx = (amax.float() / 127.0).clamp_min(1e-8)
    qx = torch.round(xh.float() / sx[:, None]).clamp_(-127, 127).to(torch.int8)
    c = torch.empty((a.shape[0], trellis.shape[1] * 16), dtype=torch.half, device="cuda")
    ok = ext.exl3_int8_gemm(qx, q, c, sx, s, False)
    assert ok
    ext.had_r_128(c, c, None, svh, 1.0)
    return c


@pytest.mark.parametrize("K", [2, 4, 6])
@pytest.mark.parametrize("mcg,mul1", CB_LIST)
def test_int8_gemm_accuracy(K, mcg, mul1):
    k, n = 1024, 1024
    trellis, suh, svh = _make(k, n, K, seed=K * 13 + (1 if mcg else 0) + (2 if mul1 else 0))
    for rows in (128, 512):
        a = torch.randn(rows, k, device="cuda").half()
        y_ref = _reference(a, trellis, suh, svh, K, mcg, mul1)
        y = _int8_gemm_path(a, trellis, suh, svh, K, mcg, mul1)
        rel = (y.float() - y_ref.float()).square().mean().sqrt() / y_ref.float().square().mean().sqrt()
        assert rel.item() < 0.02, f"K={K} m={rows} mcg={mcg} mul1={mul1}: rel rms {rel.item():.4f}"


@pytest.mark.parametrize("rows,k,n", [
    (128, 1024, 2048), (256, 3072, 1024), (512, 2048, 3072), (1, 1024, 1024),
])
def test_int8_gemm_shapes(rows, k, n):
    trellis, suh, svh = _make(k, n, 4, seed=rows + k)
    a = torch.randn(rows, k, device="cuda").half()
    y_ref = _reference(a, trellis, suh, svh, 4, False, False)
    y = _int8_gemm_path(a, trellis, suh, svh, 4, False, False)
    rel = (y.float() - y_ref.float()).square().mean().sqrt() / y_ref.float().square().mean().sqrt()
    assert rel.item() < 0.02


def test_int8_gemm_fp32_output():
    k, n = 512, 512
    trellis, suh, svh = _make(k, n, 4, seed=7)
    q, s = _weights_int8(trellis, 4, False, False)
    a = torch.randn(128, k, device="cuda").half()
    xh = torch.empty_like(a)
    ext.had_r_128(a, xh, suh, None, 1.0)
    amax = xh.abs().amax(dim=1)
    sx = (amax.float() / 127.0).clamp_min(1e-8)
    qx = torch.round(xh.float() / sx[:, None]).clamp_(-127, 127).to(torch.int8)
    c = torch.empty((128, n), dtype=torch.float32, device="cuda")
    assert ext.exl3_int8_gemm(qx, q, c, sx, s, True)
    ref = (qx.float() @ q.float().t()) * sx[:, None] * s[None, :]
    assert torch.allclose(c, ref, rtol=1e-5, atol=1e-4)


def test_int8_gemm_rejects_bad_shapes():
    q = torch.zeros((64, 96), dtype=torch.int8, device="cuda")
    c = torch.zeros((4, 64), dtype=torch.half, device="cuda")
    sx = torch.ones(4, device="cuda")
    sb = torch.ones(64, device="cuda")
    # k must be a multiple of 64
    assert not ext.exl3_int8_gemm(q, q, c, sx, sb, False)
