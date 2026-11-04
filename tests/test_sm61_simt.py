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
    """What QMatrix.reconstruct_hgemm does on its non-fused path: the stored trellis is in a
    Hadamard-rotated basis on both sides, so reconstruct gives the stored basis and the two
    rotations run as separate launches around the matmul."""
    k = trellis.shape[0] * 16
    n = trellis.shape[1] * 16
    w = torch.empty((k, n), dtype=torch.half, device="cuda")
    ext.reconstruct_slice(w, trellis, K, mcg, mul1, 0)
    xh = torch.empty_like(a)
    ext.had_r_128(a, xh, suh, None, 1.0)
    y = (xh.float() @ w.float()).to(torch.half)
    ext.had_r_128(y, y, None, svh, 1.0)
    return y


@pytest.mark.parametrize("K", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("mcg,mul1", CB_LIST)
def test_simt_gemv_matches_reconstruct(K, mcg, mul1):
    k, n = 512, 512
    trellis, suh, svh = _make(k, n, K, seed=K * 7 + (1 if mcg else 0) + (2 if mul1 else 0))
    a = torch.randn(1, k, device="cuda").half()

    y_ref = _reference(a, trellis, suh, svh, K, mcg, mul1)

    xh = torch.empty_like(a)
    ext.had_r_128(a, xh, suh, None, 1.0)
    c = torch.empty((1, n), dtype=torch.half, device="cuda")
    ext.exl3_gemm(xh, trellis, c, None, None, None, 0, mcg, mul1, 0)
    ext.had_r_128(c, c, None, svh, 1.0)
    torch.cuda.synchronize()

    rel = (c.float() - y_ref.float()).square().mean().sqrt() / y_ref.float().square().mean().sqrt()
    assert rel.item() < 0.02, f"K={K} mcg={mcg} mul1={mul1}: rel rms {rel.item():.4f}"


@pytest.mark.parametrize("K", [2, 4, 6])
def test_simt_gemv_deterministic(K):
    k, n = 512, 256
    trellis, suh, svh = _make(k, n, K, seed=99 + K)
    a = torch.randn(1, k, device="cuda").half()
    xh = torch.empty_like(a)
    ext.had_r_128(a, xh, suh, None, 1.0)
    outs = []
    for _ in range(3):
        c = torch.empty((1, n), dtype=torch.half, device="cuda")
        ext.exl3_gemm(xh, trellis, c, None, None, None, 0, False, False, 0)
        outs.append(c.clone())
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[1], outs[2])
