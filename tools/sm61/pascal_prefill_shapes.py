import os, sys, time, collections
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator
import exllamav3.modules.quant.exl3 as exl3mod

model_dir = sys.argv[1] if len(sys.argv) > 1 else "B:/git/models/qwen3-0.6b-exl3-4.0bpw"
prompt_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 2040

config = Config.from_directory(model_dir)
tokenizer = Tokenizer.from_config(config)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = 4096)
model.load(progressbar = False)

records = []
orig_recon = exl3mod.LinearEXL3.reconstruct_hgemm
orig_cb01 = exl3mod.LinearEXL3._pascal_gemm_cb01

def rec(kind, q, x):
    rows = x.numel() // x.shape[-1]
    records.append((kind, rows, q.in_features, q.out_features, q.K, q.mul1, q.mcg))

def recon(self, x, out_dtype):
    rec("recon", self, x)
    return orig_recon(self, x, out_dtype)

def cb01(self, x, out_dtype):
    rec("cb01", self, x)
    return orig_cb01(self, x, out_dtype)

exl3mod.LinearEXL3.reconstruct_hgemm = recon
exl3mod.LinearEXL3._pascal_gemm_cb01 = cb01

base = ("The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. "
        "Sphinx of black quartz, judge my vow. How vexingly quick daft zebras jump! ")
prompt = "<|im_start|>user\n" + (base * 64) + "<|im_end|>\n<|im_start|>assistant\n"
generator = Generator(model, cache, tokenizer)

generator.generate(prompt = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
                   max_new_tokens = 4, completion_only = True,
                   encode_special_tokens = True, add_bos = True)
torch.cuda.synchronize()
records.clear()

import secrets
p = prompt + f"\nRef {secrets.token_hex(8)}.<|im_end|>\n<|im_start|>assistant\n"
t0 = time.time()
generator.generate(prompt = p, max_new_tokens = 1, completion_only = True,
                   encode_special_tokens = True, add_bos = True)
torch.cuda.synchronize()
print(f"wall {time.time() - t0:.3f}s, calls {len(records)}")

by = collections.Counter((k, r, ka, n, mul1, mcg) for k, r, ka, n, K, mul1, mcg in records)
print(f"{'kind':5} {'rows':>5} {'k':>5} {'n':>6} {'mul1':>5} {'mcg':>4} {'count':>5}")
for (k, r, m, n, mul1, mcg), c in sorted(by.items(), key = lambda t: -t[0][1]):
    print(f"{k:5} {r:5} {m:5} {n:6} {mul1:5} {mcg:4} {c:5}")
