"""Measure the pre-Volta decode GEMV at 9B-class layer shapes.

Weights per token are trellis bytes = k*n*4 bits (K=4, cb0).  Bandwidth is the
number that decides decode speed on Pascal, so print it per shape and then
compose a whole synthetic token for a 36-layer dense model.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from exllamav3.ext import exllamav3_ext as ext

torch.cuda.set_device(0)
dev = "cuda:0"
K = 4


def make(k, n, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    trellis = torch.randint(0, 65536, (k // 16, n // 16, 16 * K), generator=g, dtype=torch.int32)
    trellis = trellis.to(torch.int16).to(dev)
    suh = (torch.rand(k, generator=g) * 0.5 + 0.75).to(torch.half).to(dev)
    svh = (torch.rand(n, generator=g) * 0.5 + 0.75).to(torch.half).to(dev)
    return trellis, suh, svh


def bench(k, n, iters=50, warmup=10):
    trellis, suh, svh = make(k, n)
    a = torch.randn(1, k, device=dev).half()
    xh = torch.empty_like(a)
    c = torch.empty((1, n), dtype=torch.half, device=dev)

    for _ in range(warmup):
        ext.had_r_128(a, xh, suh, None, 1.0)
        ext.exl3_gemm(xh, trellis, c, None, None, None, 0, False, False, 0)
        ext.had_r_128(c, c, None, svh, 1.0)
    torch.cuda.synchronize()

    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(iters):
        ext.had_r_128(a, xh, suh, None, 1.0)
        ext.exl3_gemm(xh, trellis, c, None, None, None, 0, False, False, 0)
        ext.had_r_128(c, c, None, svh, 1.0)
    ev1.record()
    torch.cuda.synchronize()
    full_us = ev0.elapsed_time(ev1) * 1000.0 / iters

    ev0.record()
    for _ in range(iters):
        ext.exl3_gemm(xh, trellis, c, None, None, None, 0, False, False, 0)
    ev1.record()
    torch.cuda.synchronize()
    gemm_us = ev0.elapsed_time(ev1) * 1000.0 / iters

    gb = k * n * K / 8 / 1e9
    return full_us, gemm_us, gb / (gemm_us * 1e-6)


SHAPES = [
    ("q/o proj", 4096, 4096),
    ("kv proj", 4096, 1024),
    ("gate/up", 4096, 14336),
    ("down", 14336, 4096),
    ("lm_head", 4096, 151936),
]

print(f"K={K} cb0, m=1, Tesla P40\n")
print(f"{'shape':<12} {'k':>6} {'n':>7} {'full us':>9} {'gemm us':>9} {'GB/s':>8}")
total_full = 0.0
total_bytes = 0.0
for name, k, n in SHAPES:
    full, gemm, bw = bench(k, n)
    print(f"{name:<12} {k:>6} {n:>7} {full:>9.1f} {gemm:>9.1f} {bw:>8.1f}")
    total_full += full * (2 if name in ("q/o proj", "kv proj") else 3 if name == "gate/up" or name == "down" else 1)
    total_bytes += k * n * K / 8

# Qwen3-8B-shaped: 36 layers, q,k,v,o + gate,up,down, plus lm_head
layers = 36
per_layer_us = (2 * bench(4096, 4096)[0] + 2 * bench(4096, 1024)[0]
                + 2 * bench(4096, 14336)[0] + bench(14336, 4096)[0])
lm_us = bench(4096, 151936)[0]
tok_ms = (layers * per_layer_us + lm_us) / 1000.0
print(f"\nsynthetic 36-layer token: {tok_ms:.1f} ms GPU -> {1000.0 / tok_ms:.1f} tok/s")
print(f"weights read per token: {total_bytes / 1e9 * 0:.0f}", end="")
wbytes = layers * (2 * 4096 * 4096 + 2 * 4096 * 1024 + 2 * 4096 * 14336 + 14336 * 4096) * K / 8
wbytes += 4096 * 151936 * K / 8
print(f"weight bytes/token = {wbytes / 1e9:.2f} GB")
