import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator

model_dir = sys.argv[1] if len(sys.argv) > 1 else "B:/git/models/qwen3-0.6b-exl3-4.0bpw"
prompt_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 2048

config = Config.from_directory(model_dir)
tokenizer = Tokenizer.from_config(config)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = 4096)
model.load(progressbar = False)

# Build a long prompt out of a repeating paragraph
base = (
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. "
    "Sphinx of black quartz, judge my vow. How vexingly quick daft zebras jump! "
)
prompt = "<|im_start|>user\n" + (base * 64) + "<|im_end|>\n<|im_start|>assistant\n"
ids = tokenizer.encode(prompt, encode_special_tokens = True)
while ids.numel() > prompt_tokens:
    prompt = prompt[: int(len(prompt) * 0.95)]
    ids = tokenizer.encode(prompt, encode_special_tokens = True)
prompt_len = int(ids.numel())
print(f"prompt tokens: {prompt_len}")

generator = Generator(model, cache, tokenizer)

# warm up once (allocator, kernels)
generator.generate(prompt = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
                   max_new_tokens = 4, completion_only = True,
                   encode_special_tokens = True, add_bos = True)
torch.cuda.synchronize()

t0 = time.time()
generator.generate(prompt = prompt, max_new_tokens = 1, completion_only = True,
                   encode_special_tokens = True, add_bos = True)
torch.cuda.synchronize()
t_pre = time.time() - t0
print(f"prefill wall: {t_pre:.3f}s -> {prompt_len / t_pre:.1f} tok/s")

import secrets
prompt2 = prompt + f"\nReference {secrets.token_hex(8)}.<|im_end|>\n<|im_start|>assistant\n"

t0 = time.time()
generator.generate(prompt = prompt2, max_new_tokens = 1, completion_only = True,
                   encode_special_tokens = True, add_bos = True)
torch.cuda.synchronize()
print(f"prefill (fresh prefix) wall: {time.time() - t0:.3f}s")

prompt3 = prompt + f"\nSee {secrets.token_hex(8)}.<|im_end|>\n<|im_start|>assistant\n"
from torch.profiler import profile, ProfilerActivity
t0 = time.time()
with profile(activities = [ProfilerActivity.CUDA]) as prof:
    generator.generate(prompt = prompt3, max_new_tokens = 1, completion_only = True,
                       encode_special_tokens = True, add_bos = True)
    torch.cuda.synchronize()
print(f"profiled prefill wall: {time.time() - t0:.3f}s")

evs = []
for e in prof.key_averages():
    if e.device_time_total > 0:
        evs.append((e.device_time_total, e.key, e.count))
evs.sort(reverse = True)
total = sum(t for t, _, _ in evs)
print(f"total CUDA {total / 1000:.1f} ms")
for t, k, c in evs[: 25]:
    print(f"  {t / 1000:9.2f} ms  {t / total * 100:5.1f}%  x{c:<6} {k[: 90]}")
