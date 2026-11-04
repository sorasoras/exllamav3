"""End-to-end prefill timing on the P40 with and without the int8 path.

Run with EXL3_INT8_PREFILL=1 and EXL3_INT8_PREFILL=0 and compare.
"""

import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from exllamav3 import Config, Model, Cache, Tokenizer, Generator

MODEL_DIR = "B:/git/models/qwen3-0.6b-exl3-4.0bpw"

config = Config.from_directory(MODEL_DIR)
tokenizer = Tokenizer.from_config(config)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = 8192)
model.load(progressbar = False)
generator = Generator(model, cache, tokenizer)

BASE = ("The history of computing spans mechanical calculators, relay machines, vacuum tubes, "
        "transistors and integrated circuits. ") * 60


def prompt_for(nonce):
    # nonce goes first: a differing prefix defeats the sequence-cache reuse
    return f"(run {nonce}) " + BASE


def ntok(text):
    return tokenizer.encode(text, add_bos = True).numel()


def timed(text, n_new):
    torch.cuda.synchronize()
    t0 = time.time()
    out = generator.generate(text, max_new_tokens = n_new, encode_special_tokens = True)
    torch.cuda.synchronize()
    return time.time() - t0, out


timed(prompt_for("warmup"), 2)

for trial in range(2):
    text = prompt_for(f"prefill-{trial}")
    n = ntok(text)
    dt, _ = timed(text, 1)
    print(f"prefill trial {trial}: {n} tok in {dt:.3f} s = {n / dt:.1f} tok/s")

text = prompt_for("decode")
dt, out = timed(text, 33)
print(f"decode: 32 tok in {dt:.3f} s = {32 / dt:.1f} tok/s")
if isinstance(out, (list, tuple)):
    out = out[0]
print("sample:", str(out).replace("\n", " ")[:140])
print("peak VRAM: %.2f GiB" % (torch.cuda.max_memory_allocated() / 2**30))
