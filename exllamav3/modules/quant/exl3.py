from __future__ import annotations
import torch
from ...model.config import Config
from .exl3_lib.quantize import frac_k, preapply_had_l, preapply_had_r, had_k, had_n
from ...ext import exllamav3_ext as ext
from ...util.tensor import g_tensor_cache
import os
from ...util import profile_opt

AUTO_RECONSTRUCT_THRESHOLD = 144
MAX_RECONSTRUCT_SLICE_N = 32768
RECONSTRUCT_SLICE_GRANULARITY_N = 128

no_fused_reconstruct = os.environ.get("EXL3_NO_FUSED_RECONSTRUCT", "0") != "0"

# Fused dequant-GEMM: decode the trellis into smem and multiply on the fly, so the weight
# never makes the round trip through global memory as fp16 (0.5 bytes per weight of traffic
# instead of 4). Only useful without tensor cores, i.e. on pre-Volta devices.
#
# Measured on a P40, it wins only where the fp16 copy would be far too large to stay in L2 and
# the batch is small enough that the plain FFMA path keeps up with cuBLAS: 1.17x at m=16 and
# 1.84x at m=64 for the 151936-wide lm_head, a wash from m~256 up, and 0.5-0.7x for the
# 1024-4096 wide projections (reconstruct + cuBLAS is hard to beat there: the fp16 copy is
# ~2 MB and cuBLAS runs at 9.5 of the 11.7 TFLOP/s fp32 peak, against ~6.8 for this kernel).
# So it is enabled only for the shape class where it pays off; EXL3_DQ_GEMM=1 forces it on
# everywhere, EXL3_DQ_GEMM=0 disables it.
# Unset means "auto": on, but only inside the measured win region (wide n, 8-96 rows).
# EXL3_DQ_GEMM=1 forces it everywhere (including single rows, where it is slower than the GEMV)
# and EXL3_DQ_GEMM=0 disables it.
_dq_gemm_setting = os.environ.get("EXL3_DQ_GEMM", "")
DQ_GEMM_ENABLED = _dq_gemm_setting != "0"
DQ_GEMM_FORCE = _dq_gemm_setting == "1"

# Int8 (W8A8) prefill: decode each tensor once into int8 weights with one symmetric scale per
# output row, quantize activations per row at call time, and run a dp4a GEMM (exl3_int8_gemm.cu).
# On GP102 dp4a retires 4 MACs per instruction at the FFMA issue rate -- measured 18.8 TMAC/s
# against 4.7 for FFMA on the P40 -- while cuBLAS tops out at 9.5 of the 11.7 TFLOP/s fp32 peak,
# so this is the only prefill path here that can clear the fp32 ceiling. It costs 1 byte per
# weight of extra VRAM and it is lossy, so it is opt-out and only used for rows the tensor-core
# kernels could not serve anyway. EXL3_INT8_PREFILL=0 disables it.
_int8_setting = os.environ.get("EXL3_INT8_PREFILL", "1")
INT8_PREFILL_ENABLED = _int8_setting != "0"
INT8_PREFILL_MIN_ROWS = int(os.environ.get("EXL3_INT8_PREFILL_MIN_ROWS", "128") or 128)
DQ_GEMM_MIN_N = int(os.environ.get("EXL3_DQ_GEMM_MIN_N", "16384") or 16384)
DQ_GEMM_MIN_ROWS = 8
DQ_GEMM_MAX_ROWS = int(os.environ.get("EXL3_DQ_GEMM_MAX_ROWS", "96") or 96)

class LinearEXL3:

    quant_type: str = "exl3"

    def __init__(
        self,
        config: Config | None,
        in_features: int,
        out_features: int,
        scale: torch.Tensor | None = None,
        su: torch.Tensor | None = None,
        sv: torch.Tensor | None = None,
        suh: torch.Tensor | None = None,
        svh: torch.Tensor | None = None,
        trellis: torch.Tensor | None = None,
        mcg: torch.Tensor | None = None,
        mul1: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        transformers_fix: bool = False,
        key: str | None = None
    ):
        assert scale is None, "scale is no longer used"
        assert su is not None or suh is not None, "either su (packed) or suh (unpacked) is required"
        assert sv is not None or svh is not None, "either sv (packed) or svh (unpacked) is required"
        assert trellis is not None, "trellis is required"
        if su is not None: assert su.dtype == torch.int16, "su is wrong datatype"
        if sv is not None: assert sv.dtype == torch.int16, "sv is wrong datatype"
        if suh is not None: assert suh.dtype == torch.half, "suh is wrong datatype"
        if svh is not None: assert svh.dtype == torch.half, "svh is wrong datatype"
        assert trellis.dtype == torch.int16, "trellis is wrong datatype"
        assert len(trellis.shape) == 3, "trellis must have dim = 3"

        if bias is not None and bias.dtype == torch.float: bias = bias.to(torch.half)

        # Not a Module subclass, so the config-or-NullConfig default doesn't apply here; TP imports pass
        # config=None and forward() reads config.infer_params
        if config is None:
            from ...model.config import NullConfig
            config = NullConfig()
        self.config = config
        self.transformers_fix = transformers_fix
        self.key = key

        # self.scale = scale.item()
        self.su = None
        self.sv = None
        self.suh = suh if suh is not None else self.unpack_bf(su)
        self.svh = svh if svh is not None else self.unpack_bf(sv)
        self.trellis = trellis
        # Bitrate from the tile width: 16 * K uint16 per tile, 16 * K + 8 for half-integer K (mul1 only)
        K = trellis.shape[-1] / 16
        self.K = int(K) if float(K).is_integer() else K
        self.frac = frac_k(self.K)
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.swap_device = None
        self.out_dtype = out_dtype
        self.default_out_dtype = out_dtype or torch.half

        self.mcg_tensor = mcg
        self.mul1_tensor = mul1
        self.mcg = self.mcg_tensor is not None
        self.mul1 = self.mul1_tensor is not None
        assert self.frac is None or self.mul1, f"{key}: half-integer bitrate {self.K} requires the mul1 codebook"

        self._fused_reconstruct = None
        # Pre-Volta (Pascal, sm_61) has no tensor cores: the fp16 GEMM kernels are compiled to
        # trap stubs there. Two tensor-core-free decode paths remain: the int8 dp4a GEMV for mul1
        # (exact for its codebook) and the SIMT FFMA GEMV for cb0/cb1 at a single row. Everything
        # else (prefill rows, other shapes) must go through reconstruct + cuBLAS, which is plain
        # SIMT.
        self._pre_ampere = None
        self._kernel_capable = not self.frac and self.K <= 8
        self._w8 = None
        self._w8_scale = None
        self.bsz1_xh_args = (self.trellis.device, (1, self.in_features), self.out_dtype)
        # K is the bitrate (int, or float for the half-integer rates); the C++ side decomposes it
        self.bc = ext.BC_LinearEXL3(
            self.trellis,
            self.suh,
            self.svh,
            self.K,
            self.bias,
            self.mcg,
            self.mul1,
            g_tensor_cache.get(*self.bsz1_xh_args)
        )


    def unload(self):
        # g_tensor_cache.drop(*self.bsz1_xh_args)
        pass


    def get_tensors(self, key: str):
        return {
            f"{key}.{subkey}": tensor.contiguous()
            for subkey, tensor in [
                ("su", self.su),
                ("sv", self.sv),
                ("suh", self.suh),
                ("svh", self.svh),
                ("trellis", self.trellis),
                ("bias", self.bias),
                ("mcg", self.mcg_tensor),
                ("mul1", self.mul1_tensor),
            ] if tensor is not None
        }


    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        if "ovr" in params:
            ovr = params["ovr"]
            if self.key in ovr and ovr[self.key].inner is not self:
                return ovr[self.key].forward(x, params, out_dtype)

        # The EXL3 kernels read x as contiguous rows; a strided view (e.g. a head-group slice
        # of a wider tensor) would be silently misread as interleaved garbage. Producers are
        # responsible for contiguity (a silent copy here would hide a hot-path inefficiency
        # and break CUDA-graph address stability)
        assert x.is_contiguous(), f"LinearEXL3 {self.key}: non-contiguous input {tuple(x.shape)}"

        reconstruct = params.get("reconstruct")
        if not reconstruct:
            rows = x.numel() // x.shape[-1]
            if self._is_pre_ampere():
                if not self._pascal_kernel_ok(rows):
                    # No tensor cores here and neither tensor-core-free GEMV can take this call,
                    # so the kernel path would trap. Reconstruct regardless of the row count.
                    return self.reconstruct_hgemm(x, out_dtype)
                if not self.mul1:
                    return self._pascal_gemm_cb01(x, out_dtype)
            if rows <= AUTO_RECONSTRUCT_THRESHOLD or self.config.infer_params.no_reconstruct:
                dtype = out_dtype or self.default_out_dtype
                return self.bc.run_alloc(x, self.out_features, dtype == torch.float)

        return self.reconstruct_hgemm(x, out_dtype)


    def _pascal_gemm_cb01(self, x: torch.Tensor, out_dtype):
        # EXL3 Hadamard-transforms both sides of a tensor at quant time, so the consumer applies
        # the two matching rotations. The Ampere+ kernels fold them into the launch; on Pascal
        # they are separate launches around a plain dot product.
        #
        # One interpreter crossing per projection matters: a decode step runs ~200 projections,
        # and four torch calls plus two scratch allocations each cost more than the kernels
        # themselves (8.4 ms of GPU work against 34 ms wall per token).
        shape = x.shape
        x = x.reshape(1, self.in_features)
        out_dtype = out_dtype or self.default_out_dtype
        y = torch.empty(shape[:-1] + (self.out_features,), dtype=out_dtype, device=x.device)
        if ext.exl3_gemv_cb01(x, self.trellis, y.view(1, self.out_features), self.suh, self.svh,
                              self.bias, self.K, self.mcg, self.mul1):
            return y
        return self.reconstruct_hgemm(x, out_dtype)

    def _try_dequant_gemm(self, x: torch.Tensor, y_: torch.Tensor) -> bool:
        # exl3_dequant_gemm decodes the trellis into shared memory and multiplies there, in the
        # rotated basis like the non-fused reconstruct path: rotate the input, GEMM, rotate the
        # output back. Returns False (leaving y_ untouched) when the shape, rank or device is not
        # covered, so the caller can fall back to reconstruct + cuBLAS.
        if x.shape[1] % 16 or y_.shape[1] % 128:
            return False
        xh = torch.empty_like(x)
        ext.had_r_128(x, xh, self.suh, None, 1.0)
        if not ext.exl3_dequant_gemm(xh, self.trellis, y_, self.K, self.mcg, self.mul1):
            return False
        ext.had_r_128(y_, y_, None, self.svh, 1.0)
        return True


    def _int8_weights(self):
        # One-time decode of the trellis into int8 [out_features, in_features] with a symmetric
        # scale per output row, in the same stored basis the non-fused reconstruct path uses, so
        # the caller rotates input and output exactly like that path. Built in column chunks to
        # avoid a full fp32 temporary. Returns (None, None) when the tensor is not covered or the
        # copy does not fit, and remembers the refusal so the fallback is not retried per call.
        if self._w8 is False:
            return None, None
        if self._w8 is not None:
            return self._w8, self._w8_scale

        dev = self.trellis.device
        k, n = self.in_features, self.out_features
        chunk = 2048
        q = None
        try:
            with torch.cuda.device(dev):
                free, _ = torch.cuda.mem_get_info(dev)
                if free < n * k * 1.1:
                    raise RuntimeError("insufficient free VRAM for the int8 weight copy")
                q = torch.empty((n, k), dtype=torch.int8, device=dev)
                scale = torch.empty((n,), dtype=torch.float32, device=dev)
                for n0 in range(0, n, chunk):
                    nc = min(chunk, n - n0)
                    w = torch.empty((k, nc), dtype=torch.half, device=dev)
                    self._reconstruct_inner(w, n0)
                    wf = w.float()
                    amax = wf.abs().amax(dim=0)
                    s = (amax / 127.0).clamp_min(1e-8)
                    q[n0:n0 + nc] = torch.round(wf / s).clamp_(-127, 127).to(torch.int8).t()
                    scale[n0:n0 + nc] = s
                    del w, wf
        except RuntimeError:
            self._w8 = False
            return None, None

        self._w8 = q
        self._w8_scale = scale
        return q, scale


    def _int8_prefill(self, x: torch.Tensor, y_: torch.Tensor) -> bool:
        w8, w8s = self._int8_weights()
        if w8 is None:
            return False

        xh = torch.empty_like(x)
        ext.had_r_128(x, xh, self.suh, None, 1.0)
        qx = torch.empty(x.shape, dtype=torch.int8, device=x.device)
        sx = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
        ext.exl3_quantize_rows_int8(xh, qx, sx)

        if y_.dtype == torch.half:
            if not ext.exl3_int8_gemm(qx, w8, y_, sx, w8s, False):
                return False
            ext.had_r_128(y_, y_, None, self.svh, 1.0)
            return True

        yh = torch.empty(y_.shape, dtype=torch.half, device=x.device)
        if not ext.exl3_int8_gemm(qx, w8, yh, sx, w8s, False):
            return False
        ext.had_r_128(yh, yh, None, self.svh, 1.0)
        y_.copy_(yh.float())
        return True


    def _is_pre_ampere(self) -> bool:
        if self._pre_ampere is None:
            d = self.trellis.device
            self._pre_ampere = d.type == "cuda" and torch.cuda.get_device_capability(d)[0] < 7
        return self._pre_ampere


    def _pascal_kernel_ok(self, rows: int) -> bool:
        # Mirrors the shape limits of the two tensor-core-free GEMVs. mul1 goes to the int8 dp4a
        # dispatcher (exl3_gemv_int8.cu): the cooperative kernel is single-row, the shared-memory
        # path takes m <= 2, and only the cooperative kernel has K = 7/8 instances. cb0/cb1 go to
        # the SIMT GEMV (exl3_gemv_simt.cu), which is single-row by construction.
        if not self._kernel_capable:
            return False
        if not self.mul1:
            return rows == 1
        try:
            mode = int(os.environ.get("EXL3_INT8_GEMV", "2") or 0)
        except ValueError:
            mode = 2
        rows_max = 0 if mode == 0 else (1 if mode == 1 else 2)
        if rows > rows_max:
            return False
        return rows <= 1 or self.K <= 6


    def unpack_bf(self, bitfield: torch.Tensor):
        # For some reason this operation causes a GPU assert on Transformers. Running on CPU seems to fix it
        device = bitfield.device
        if self.transformers_fix:
            bitfield = bitfield.cpu()

        # (Only used for full reconstruct and loading old models, not during inference)
        bitfield = bitfield.view(torch.uint16).to(torch.int)
        masks = (1 << torch.arange(16)).to(bitfield.device)
        expanded = (bitfield.unsqueeze(-1) & masks) > 0
        expanded = expanded.flatten()
        # NOT torch.where with CPU scalar tensors: that path misses the device guard when the
        # condition lives on a non-current device (observed on torch 2.11.0+cu130) — the kernel
        # launches on the current device's context, faults there, silently zero-fills the output
        # and leaves every other device in the process unusable. Map bool -> {-1, +1} arithmetically
        expanded = 1.0 - expanded.to(torch.float16) * 2.0
        return expanded.contiguous().to(device)


    def reconstruct_hgemm(self, x: torch.Tensor, out_dtype):

        shape = x.shape
        rows = x.numel() // shape[-1]
        out_shape = shape[:-1] + (self.out_features,)
        x = x.view(rows, self.in_features)
        y = torch.empty(out_shape, dtype = out_dtype or self.default_out_dtype, device = x.device)

        y_ = y.view(rows, self.out_features)

        # Int8 prefill: pre-quantized weights, dp4a GEMM. Tried first because it is strictly
        # faster than the fused dequant-GEMM where it applies (no trellis decode in the inner
        # loop); the weight copy is built lazily on the first call that reaches this point.
        if (INT8_PREFILL_ENABLED and rows >= INT8_PREFILL_MIN_ROWS and not self.frac
                and self._is_pre_ampere()
                and x.shape[1] % 64 == 0 and self.out_features % 64 == 0):
            if self._int8_prefill(x, y_):
                if self.bias is not None:
                    y += self.bias
                return y

        # Fused dequant-GEMM: the trellis slab is decoded in shared memory and consumed there,
        # so the weight traffic is the packed trellis itself instead of a materialized fp16
        # copy that is written once and read once (4 bytes per weight). It runs in the rotated
        # basis, so it owns both had_r_128 launches. Pre-Volta only: tensor-core devices are
        # far better served by the regular gemm kernels.
        if (DQ_GEMM_ENABLED and 0 < rows <= DQ_GEMM_MAX_ROWS and self._is_pre_ampere()
                and (DQ_GEMM_FORCE or (rows >= DQ_GEMM_MIN_ROWS and self.out_features >= DQ_GEMM_MIN_N))):
            if self._try_dequant_gemm(x, y_):
                if self.bias is not None:
                    y += self.bias
                return y

        # Fused path: reconstruct emits ORIGINAL-basis weights (both Hadamards + sign
        # vectors folded into the memory-bound reconstruct kernel), so the gemm runs on the
        # raw input and the standalone input/output had_r_128 launches disappear (~14% of
        # long-chunk prefill GPU time). Requires 128-divisible dims (always true for EXL3
        # tensors: both sides are had-transformed at quant time)
        if self._fused_reconstruct is None:
            self._fused_reconstruct = (
                self.in_features % 128 == 0 and self.out_features % 128 == 0
                and not no_fused_reconstruct
            )

        # The fused kernel costs ~4x plain reconstruct (k*n-proportional) while the saved
        # had launches scale with rows*(k+n); breakeven is rows ~400-900 across shapes
        use_fused = self._fused_reconstruct and rows >= 1024

        if use_fused:
            xh = x
        else:
            xh = torch.empty_like(x)
            ext.had_r_128(x, xh, self.suh, None, 1.0)

        if self.out_features <= MAX_RECONSTRUCT_SLICE_N:
            w = torch.empty((self.in_features, self.out_features), dtype = torch.half, device = self.trellis.device)
            if use_fused:
                ext.reconstruct_had_slice(w, self.trellis, self.suh, self.svh, self.K, self.mcg, self.mul1, 0)
            else:
                self._reconstruct_inner(w, 0)
            ext.hgemm_recon(xh, w, y_)
        else:
            numel_ = self.in_features * MAX_RECONSTRUCT_SLICE_N
            w_ = torch.empty((numel_,), dtype = torch.half, device = self.trellis.device)
            for n_start in range(0, self.out_features, MAX_RECONSTRUCT_SLICE_N):
                n_end = min(n_start + MAX_RECONSTRUCT_SLICE_N, self.out_features)
                numel = self.in_features * (n_end - n_start)
                w = w_[:numel].view(self.in_features, n_end - n_start)
                if use_fused:
                    ext.reconstruct_had_slice(
                        w, self.trellis, self.suh, self.svh[n_start:], self.K, self.mcg, self.mul1, n_start)
                else:
                    self._reconstruct_inner(w, n_start)
                ext.hgemm_recon(xh, w, y_[:, n_start:n_end])

        if not use_fused:
            ext.had_r_128(y_, y_, None, self.svh, 1.0)

        if self.bias is not None:
            y += self.bias
        return y


    def _reconstruct_inner(self, w: torch.Tensor, n_start: int):
        """Rotated-basis weight slice (in_features, w.shape[1]) from packed columns n_start.."""
        if n_start == 0 and w.shape[1] == self.out_features:
            ext.reconstruct(w, self.trellis, self.K, self.mcg, self.mul1)
        else:
            ext.reconstruct_slice(w, self.trellis, self.K, self.mcg, self.mul1, n_start)

    def get_inner_weight_tensor(self, n_offset: int = 0, n_features: int | None = None):
        w = torch.empty((self.in_features, self.out_features), dtype = torch.half, device = self.trellis.device)
        self._reconstruct_inner(w, 0)
        return w


    def get_weight_tensor(self):
        # suh = self.unpack_bf(self.su).unsqueeze(1)
        suh = self.unpack_bf(self.su).unsqueeze(1) if self.su else self.suh.unsqueeze(1)
        svh = self.unpack_bf(self.sv).unsqueeze(0) if self.sv else self.svh.unsqueeze(0)
        w = self.get_inner_weight_tensor()
        w = preapply_had_l(w, had_k)
        w *= suh
        w = preapply_had_r(w, had_n)
        w *= svh
        # w *= self.scale
        return w


    def get_bias_tensor(self) -> torch.Tensor | None:
        return self.bias


    # Swap tensors to CPU (to free some space while quantizing)
    def swap_cpu(self):
        if self.swap_device is not None:
            return
        self.swap_device = self.trellis.device
        if self.su is not None: self.su = self.su.cpu()
        if self.sv is not None: self.sv = self.sv.cpu()
        if self.suh is not None: self.suh = self.suh.cpu()
        if self.svh is not None: self.svh = self.svh.cpu()
        if self.trellis is not None: self.trellis = self.trellis.cpu()
        if self.bias is not None: self.bias = self.bias.cpu()


    def unswap_cpu(self):
        if self.swap_device is None:
            return
        if self.su is not None: self.su = self.su.to(self.swap_device)
        if self.sv is not None: self.sv = self.sv.to(self.swap_device)
        if self.suh is not None: self.suh = self.suh.to(self.swap_device)
        if self.svh is not None: self.svh = self.svh.to(self.swap_device)
        if self.trellis is not None: self.trellis = self.trellis.to(self.swap_device)
        if self.bias is not None: self.bias = self.bias.to(self.swap_device)
        self.swap_device = None


    def tp_export(self, plan, producer):
        return {
            "cls": LinearEXL3,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "suh": producer.send(self.suh),
            "svh": producer.send(self.svh),
            "trellis": producer.send(self.trellis),
            "bias": producer.send(self.bias),
            "mcg": producer.send(self.mcg_tensor),
            "mul1": producer.send(self.mul1_tensor),
            "out_dtype": self.out_dtype,
        }


    @staticmethod
    def tp_import_split(local_context, exported, plan, split):
        consumer = local_context["consumer"]
        device = local_context["device"]
        id_suh = exported["suh"]
        id_svh = exported["svh"]
        id_trellis = exported["trellis"]
        id_bias = exported["bias"]
        mcg = consumer.recv(exported["mcg"], cuda = True)
        mul1 = consumer.recv(exported["mul1"], cuda = True)

        if split is not None:
            split_out, first, last = split
        else:
            split_out, first, last = True, 0, exported["out_features"]

        if split_out:
            suh = consumer.recv(id_suh, cuda = True)
            svh = consumer.recv(id_svh, cuda = True, slice_dim = 0, first = first, last = last)
            trellis = consumer.recv(id_trellis, cuda = True, slice_dim = 1, first = first // 16, last = last // 16)
            bias = consumer.recv(id_bias, cuda = True, slice_dim = 0, first = first, last = last)
            in_features = exported["in_features"]
            out_features = last - first
        else:
            suh = consumer.recv(id_suh, cuda = True, slice_dim = 0, first = first, last = last)
            svh = consumer.recv(id_svh, cuda = True)
            trellis = consumer.recv(id_trellis, cuda = True, slice_dim = 0, first = first // 16, last = last // 16)
            bias = consumer.recv(id_bias, cuda = True) if first == 0 else None
            in_features = last - first
            out_features = exported["out_features"]

        module = LinearEXL3(
            config = None,
            in_features = in_features,
            out_features = out_features,
            scale = None,
            su = None,
            sv = None,
            suh = suh,
            svh = svh,
            trellis = trellis,
            mcg = mcg,
            mul1 = mul1,
            bias = bias,
            out_dtype = exported["out_dtype"],
        )
        return module


    @staticmethod
    def tp_import_split_3(local_context, exported, plan, split_0, split_1, split_2, dbg = False):
        return LinearEXL3.tp_import_split_n(local_context, exported, plan, [split_0, split_1, split_2], dbg)


    @staticmethod
    def tp_import_split_n(local_context, exported, plan, splits, dbg = False):
        consumer = local_context["consumer"]
        device = local_context["device"]
        id_suh = exported["suh"]
        id_svh = exported["svh"]
        id_trellis = exported["trellis"]
        id_bias = exported["bias"]
        mcg = consumer.recv(exported["mcg"], cuda = True)
        mul1 = consumer.recv(exported["mul1"], cuda = True)

        svh_ = []
        trellis_ = []
        bias_ = []
        in_features = 0
        out_features = 0

        for split in splits:
            assert split is not None
            split_out, first, last = split
            assert split_out

            suh = consumer.recv(id_suh, cuda = True)
            svh = consumer.recv(id_svh, cuda = True, slice_dim = 0, first = first, last = last)
            trellis = consumer.recv(id_trellis, cuda = True, slice_dim = 1, first = first // 16, last = last // 16)
            bias = consumer.recv(id_bias, cuda = True, slice_dim = 0, first = first, last = last)
            in_features = exported["in_features"]
            out_features += last - first
            svh_.append(svh)
            trellis_.append(trellis)
            bias_.append(bias)

        svh = torch.cat(svh_, dim = 0)
        trellis = torch.cat(trellis_, dim = 1)
        bias = torch.cat(bias_, dim = 0) if bias_[0] is not None else None

        module = LinearEXL3(
            config = None,
            in_features = in_features,
            out_features = out_features,
            scale = None,
            su = None,
            sv = None,
            suh = suh,
            svh = svh,
            trellis = trellis,
            mcg = mcg,
            mul1 = mul1,
            bias = bias,
            out_dtype = exported["out_dtype"],
        )
        return module
