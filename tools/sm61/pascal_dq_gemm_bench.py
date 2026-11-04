import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from exllamav3.ext import exllamav3_ext as ext

torch.cuda.set_device(0)


def make(k, n, K, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.randint(0, 65536, (k // 16, n // 16, 16 * K), generator=g, dtype=torch.int32)
    t = t.to(torch.int16).cuda()
    suh = (torch.rand(k, generator=g) * 0.5 + 0.75).half().cuda()
    svh = (torch.rand(n, generator=g) * 0.5 + 0.75).half().cuda()
    return t, suh, svh


def bench(fn, iters):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def run(m, k, n, K=4):
    trellis, suh, svh = make(k, n, K, seed=m * 7 + k + n)
    a = torch.randn(m, k, device="cuda").half()
    xh = torch.empty_like(a)
    xh2 = torch.empty_like(a)
    y = torch.empty((m, n), dtype=torch.half, device="cuda")
    w = torch.empty((k, n), dtype=torch.half, device="cuda")

    def fuse():
        ext.had_r_128(a, xh, suh, None, 1.0)
        ext.exl3_dequant_gemm(xh, trellis, y, K, False, False)
        ext.had_r_128(y, y, None, svh, 1.0)

    def recon():
        ext.had_r_128(a, xh2, suh, None, 1.0)
        ext.reconstruct_slice(w, trellis, K, False, False, 0)
        ext.hgemm_recon(xh2, w, y)
        ext.had_r_128(y, y, None, svh, 1.0)

    # gemm only, with the rotations excluded, to isolate the projection itself
    def fuse_gemm():
        ext.exl3_dequant_gemm(xh, trellis, y, K, False, False)

    def recon_gemm():
        ext.reconstruct_slice(w, trellis, K, False, False, 0)
        ext.hgemm_recon(xh, w, y)

    iters = max(8, min(400, int(4e8 // (m * k + k * n + m * n))))
    tf = bench(fuse, iters)
    tr = bench(recon, iters)
    tfg = bench(fuse_gemm, iters)
    trg = bench(recon_gemm, iters)
    wbytes = k * n * (K * 4.0 / 8.0) / 1e9  # packed trellis size in GB
    print(
        f"m={m:5d} k={k:5d} n={n:6d} | fused {tf * 1e3:8.1f}us  recon {tr * 1e3:8.1f}us  "
        f"x{tr / tf:4.2f} | gemm-only {tfg * 1e3:8.1f} vs {trg * 1e3:8.1f} "
        f"({wbytes / (tfg * 1e-3):6.1f} vs {wbytes / (trg * 1e-3):6.1f} GB/s trellis)"
    )


print(torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0))
for (k, n) in [(1024, 1024), (2048, 1024), (3072, 3072)]:
    for m in [1, 16, 64, 256, 477, 1024, 2045]:
        run(m, k, n)

print("lm_head-like:")
for m in [1, 16, 64, 477]:
    run(m, 1024, 151936)
