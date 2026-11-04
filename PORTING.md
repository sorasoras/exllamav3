# Pascal (sm_61) support

This branch adds a working path for pre-Volta NVIDIA GPUs, developed and measured on a
single **Tesla P40 (compute capability 6.1)**. Upstream exllamav3 targets Volta and later:
`mma.sync`, `ldmatrix`, `cp.async`, scoped acquire/release and `__nanosleep` all require
sm_70 or sm_75, so before this branch the extension did not compile for sm_61 at all.

Status: **experimental**. Generation is correct and reasonably fast for a 2016 card, but the
port misses the fast paths that make exllamav3 quick on modern hardware (graph-captured
decode, tensor-core prefill). See [Limitations](#limitations).

---

## Quick start

Requirements:

* A Pascal GPU (sm_60/sm_61; tested on sm_61 only)
* CUDA 12.x — **12.6 is the last toolkit that still compiles below sm_75**
* PyTorch built for CUDA 12.6 (`torch==2.8.0+cu126` was used). cu128 and later wheels no
  longer contain Pascal kernels.
* MSVC 2019/2022 for the Windows build

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python -m pip install -e .
tools/sm61/build.bat            # rebuilds the extension with TORCH_CUDA_ARCH_LIST=6.1
```

`tools/sm61/build.bat` enters the Visual Studio developer environment itself, because PyTorch
rejects a cl.exe newer than the versions it was built against.

Then:

```bash
python -m pytest tests/test_sm61_pascal.py tests/test_sm61_simt.py \
                 tests/test_sm61_dequant_gemm.py tests/test_sm61_int8_gemm.py -q
python tools/sm61/pascal_e2e.py <model_dir> [max_new_tokens]
```

---

## How it works

Everything ships in one fatbin. A separate pre-Volta build is not an option: a single process
has to be able to see a Pascal card next to an Ampere+ card, and one `.pyd` must serve both.
So the arch selection happens **inside the device code**, via `#if` gating, and kernel entry
points that cannot exist on Pascal compile to `__trap()` stubs instead of invalid PTX.

### Arch macros (`exllamav3_ext/arch.cuh`)

| macro | meaning |
| --- | --- |
| `EXL3_SM75` | pre-Ampere (`__CUDA_ARCH__ < 800`) — unchanged meaning, so the existing `cp.async` fallbacks already cover Pascal |
| `EXL3_SM61` | pre-Volta (`__CUDA_ARCH__ < 700`) — the new gate, for instructions that need Volta |
| `EXL3_NANOSLEEP(ns)` | `__nanosleep` on sm_70+, no-op on sm_61 |
| `EXL3_GRID_CONSTANT` | `__grid_constant__` on sm_70+, empty on sm_61 (it is a Volta feature) |

### Tensor-core code paths

`ptx.cuh` keeps the m16n8k16 / m8n8k4 mma wrappers, `ldmatrix`, and the scoped
acquire/release barriers for Volta and later. On sm_61 they compile to `(void)` casts plus
`__trap()`: the block GEMM, MoE kernels and fp16 GEMV are simply never dispatched there. The
acquire/release helpers get a real fallback instead of a trap (volatile load + `__threadfence`)
because the parallel collectives are built on them.

`exl3_gemm()` and `exl3_mgemm()` raise a clear `TORCH_CHECK` on a pre-Volta device rather than
launching a stub, so a mis-routed call fails loudly instead of trapping.

### Decode: `exllamav3_ext/quant/exl3_gemv_simt.cu`

The only tensor-core-free decode kernel. One warp decodes a 16-column tile with the same
`dq_dispatch` used by `reconstruct`, then a column reduction over the 4 lanes that share a
column (`__shfl_xor_sync` 1 then 2). Grid is `(n_tiles / 8, split)`: split-K exists because
`n/16` warps alone cannot fill the SMs, and decode is latency-bound, not bandwidth-bound.

Codebook math is where Pascal hurts: GP102 executes fp16 arithmetic at 1/64 of fp32. The
`exl3_hadd` / `exl3_hadd2` / `exl3_hfma2` / `exl3_hfma` wrappers in `codebook.cuh` do the
per-weight add/fma in fp32 on pre-Volta and in native half otherwise. This alone was 2.6x on
the decode GEMV. The fp32 result is bit-identical for the adds and within 1 ulp for the fma.

`exl3_gemv_cb01()` wraps the two Hadamard rotations, the GEMV and the bias into one call, since
EXL3 stores weights in a Hadamard-rotated basis on both sides.

### Prefill: reconstruct, fused dequant, or int8

Three implementations, chosen by shape:

1. **`reconstruct` + cuBLAS** (`modules/quant/exl3.py`): dequantize to fp16 and call cuBLAS.
   The default, and the fallback for everything.
2. **`exl3_dequant_gemm`**: dequantizes tiles into shared memory and never materializes the
   fp16 weight matrix. Enabled for `8 <= rows <= 96` with `out_features >= 16384`, which is
   the only region where it beats cuBLAS (1.28-1.84x on wide `lm_head`-shaped tensors). Below
   that it loses, because a 1024-3072 wide fp16 copy stays in L2 and the extra traffic never
   reaches DRAM.
3. **`exl3_int8_gemm`**: requantizes weights to int8 once (1 B/weight, lazy, per output row)
   and runs a dp4a GEMM. On the P40 dp4a issues at 4x the fp32 FMA rate (18.8 vs 4.7 TMAC/s,
   measured), so this is the fastest prefill path, at the cost of ~1% relative error and one
   extra byte per weight of VRAM. The kernel sits at 57% of the dp4a peak; the tile schedule
   is L2-swizzled because an n-fastest block order saturated DRAM (328 of 346 GB/s) by
   re-reading the same weight columns from all 32 m-tiles.

### Python-side routing (`modules/quant/exl3.py`)

`LinearEXL3` picks a path by row count when the tensor lives on a pre-Volta device:

| rows | path |
| --- | --- |
| 1 | `exl3_gemv_cb01` (fused GEMV) for any codebook with K ≤ 8 |
| 1-2, mul1 | existing int8 dp4a GEMV (K up to 8 on Pascal) |
| 8-96, wide n | fused dequant GEMM |
| ≥ 128 | int8 prefill: row quantizer + dp4a GEMM |
| otherwise | reconstruct + cuBLAS |

Fused MGEMM projections, BC attention and BC MLP are disabled per device
(`InferParams.use_mgemm(device)`, `bc_attn._capable_device`, `mlp._no_graph_path`) because all
of them are mma- or graph-based. Triton kernels are filtered out of the attention dispatch on
pre-Volta rather than deprioritized: Triton's CUDA backend JIT-fails there instead of returning
None. A `fn_torch_paged_attn_fallback` was added so paged attention works at any head dim with
plain SDPA.

### Environment variables

| variable | default | effect |
| --- | --- | --- |
| `EXL3_INT8_GEMV` | `2` | `0` off, `1` residual error feedback (~15-16 bit), `2` plain int8 |
| `EXL3_INT8_GEMV_MAX_K` | cc-based | how many bits the dp4a GEMV handles (8 on Pascal) |
| `EXL3_DQ_GEMM` | unset (auto) | `1` forces the fused dequant GEMM everywhere, `0` disables |
| `EXL3_DQ_GEMM_MIN_N` / `_MAX_ROWS` | `16384` / `96` | its dispatch window |
| `EXL3_INT8_PREFILL` | `1` | int8 prefill path on/off |
| `EXL3_INT8_PREFILL_MIN_ROWS` | `128` | row count where it takes over |

---

## Measurements (Tesla P40, Qwen3-0.6B-exl3 4.0bpw)

| | value |
| --- | --- |
| decode | 30-39 tok/s (prompt dependent) |
| prefill | ~4400 tok/s at 1327 tokens |
| decode GEMV | 12-14 µs per 1024x1024 projection (92-140 GB/s) |
| int8 prefill GEMM | 57% of the 18.8 TMAC/s dp4a peak |
| test suite | 122 tests |

A 9B at 4 bpw reads 4.24 GB of weights per token and the GEMV sustains 85-140 GB/s across its
layer shapes, so ~24 tok/s is the ceiling at short context (~11-13 at 32k, where the KV cache
adds 4.7 GB of reads per token).

Both phases are currently **host-bound**: a 1327-token prefill sums to 22 ms of kernel time
inside a 300 ms wall, and decode is 13.8 ms of GPU against 34 ms of wall (20 ms/token of CPU,
`cudaLaunchKernel` alone 4.65 ms). The graph-captured BC path that hides this on modern cards
is mma-based and disabled here.

---

## Limitations

* No CUDA graph capture for decode — the largest remaining win (~2x).
* Attention runs through the SDPA fallback. That is ~70% of prefill kernel time and the
  obvious place for a native paged-attention kernel.
* No int8 dp4a path for prefill shapes above ~96 rows on the `cb0`/`cb1` codebooks: those
  codebooks are not byte-affine, so they cannot be folded into `dp4a` the way `mul1` can.
  The int8 prefill path requantizes instead, which is lossy.
* MoE expert GEMM (`exl3_mgemm`) is mma-only and raises on Pascal.
* A single P40 was used for all validation; mixed Pascal + Ampere rigs are untested, though
  the routing decisions are per-device and nothing in the gating assumes one architecture.

## Layout of this branch

```
exllamav3_ext/arch.cuh                  EXL3_SM61, EXL3_NANOSLEEP, EXL3_GRID_CONSTANT
exllamav3_ext/ptx.cuh                   mma/ldmatrix/barrier/atomic gating
exllamav3_ext/quant/codebook.cuh        fp32 codebook math on pre-Volta
exllamav3_ext/quant/exl3_gemv_simt.*    tensor-core-free decode GEMV (new)
exllamav3_ext/quant/exl3_dequant_gemm.* fused dequant GEMM (new)
exllamav3_ext/quant/exl3_int8_gemm.*    int8 dp4a GEMM + row quantizer (new)
exllamav3_ext/quant/exl3_gemv_int8*     sm_61 dispatch, smem budgets, K <= 8
exllamav3_ext/quant/exl3_devctx.*       CC_PASCAL
modules/quant/exl3.py                   row-count routing, fused projections
modules/attention_fn/{dispatch,torch}.py  Triton filtering, paged SDPA fallback
tools/sm61/                              build script, benchmarks, e2e harness
tests/test_sm61_*.py                     122 tests
```
