"""Int8 prefill GEMM vs reconstruct + cuBLAS at prefill shapes, on the P40.

Both paths include the two had_r_128 rotations, since that is what the model pays.
int8 additionally pays the per-row activation quantization.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from exllamav3.ext import exllamav3_ext as ext

dev = "cuda:0"
K = 4


def make(k, n, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    t = torch.randint(0, 65536, (k // 16, n // 16, 16 * K), generator=g, dtype=torch.int32)
    return t.to(torch.int16).to(dev)


def build_int8(trellis, k, n, chunk=2048):
    q = torch.empty((n, k), dtype=torch.int8, device=dev)
    s = torch.empty((n,), dtype=torch.float32, device=dev)
    for n0 in range(0, n, chunk):
        nc = min(chunk, n - n0)
        w = torch.empty((k, nc), dtype=torch.half, device=dev)
        ext.reconstruct_slice(w, trellis, K, False, False, n0)
        wf = w.float()
        amax = wf.abs().amax(dim=0)
        sc = (amax / 127.0).clamp_min(1e-8)
        q[n0:n0 + nc] = torch.round(wf / sc).clamp_(-127, 127).to(torch.int8).t()
        s[n0:n0 + nc] = sc
    return q, s


def timeit(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters


def run(m, k, n):
    trellis = make(k, n)
    suh = torch.rand(k, device=dev).mul(0.5).add(0.75).half()
    svh = torch.rand(n, device=dev).mul(0.5).add(0.75).half()
    a = torch.randn(m, k, device=dev).half()
    q, s = build_int8(trellis, k, n)
    yh = torch.empty((m, n), dtype=torch.half, device=dev)

    def recon():
        xh = torch.empty_like(a)
        ext.had_r_128(a, xh, suh, None, 1.0)
        w = torch.empty((k, n), dtype=torch.half, device=dev)
        ext.reconstruct(w, trellis, K, False, False)
        ext.hgemm_recon(xh, w, yh)
        ext.had_r_128(yh, yh, None, svh, 1.0)

    def int8():
        xh = torch.empty_like(a)
        ext.had_r_128(a, xh, suh, None, 1.0)
        amax = xh.abs().amax(dim=1)
        sx = (amax.float() / 127.0).clamp_min(1e-8)
        qx = torch.round(xh.float() / sx[:, None]).clamp_(-127, 127).to(torch.int8)
        ext.exl3_int8_gemm(qx, q, yh, sx, s, False)
        ext.had_r_128(yh, yh, None, svh, 1.0)

    t_r = timeit(recon)
    t_i = timeit(int8)
    tf = 2 * m * k * n / 1e12
    print(f"m={m:<5} k={k:<6} n={n:<7} recon {t_r:8.2f} ms ({tf / (t_r / 1e3):5.2f} TFLOP/s)   "
          f"int8 {t_i:8.2f} ms ({tf / (t_i / 1e3):5.2f} TFLOP/s eq)   {t_r / t_i:5.2f}x")
    del q, s, trellis, a, yh
    torch.cuda.empty_cache()


print(f"K={K} cb0, GEMV/GEMM only (bias excluded), P40\n")
for m, k, n in [
    (128, 1024, 3072),
    (2045, 1024, 3072),
    (2045, 4096, 4096),
    (2045, 4096, 14336),
    (2045, 14336, 4096),
    (2045, 4096, 151936),
]:
    run(m, k, n)
