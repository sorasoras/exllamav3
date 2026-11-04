import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator

model_dir = sys.argv[1] if len(sys.argv) > 1 else "B:/git/models/qwen3-0.6b-exl3-4.0bpw"
max_new = int(sys.argv[2]) if len(sys.argv) > 2 else 64

print("torch", torch.__version__, "cuda", torch.version.cuda, "device", torch.cuda.get_device_name(0))

config = Config.from_directory(model_dir)
tokenizer = Tokenizer.from_config(config)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = 4096)
model.load(progressbar = False)
print("model loaded, cc:", torch.cuda.get_device_capability(0))
generator = Generator(model, cache, tokenizer)

prompt = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\nName the three primary colors, in one short sentence.<|im_end|>\n"
    "<|im_start|>assistant\n"
)
input_ids = tokenizer.encode(prompt, add_bos = True, encode_special_tokens = True)

# Prefill
t0 = time.time()
out = generator.generate(
    prompt = prompt,
    max_new_tokens = 1,
    completion_only = True,
    encode_special_tokens = True,
    add_bos = True,
)
torch.cuda.synchronize()
t_prefill = time.time() - t0
prompt_len = int(input_ids.numel()) if hasattr(input_ids, "numel") else len(input_ids)
print(f"prefill ({prompt_len} tok): {t_prefill:.2f}s  -> {prompt_len/t_prefill:.1f} tok/s")
print("first token:", repr(out))

# Decode
t0 = time.time()
out = generator.generate(
    prompt = prompt,
    max_new_tokens = max_new,
    completion_only = True,
    encode_special_tokens = True,
    add_bos = True,
)
torch.cuda.synchronize()
t_gen = time.time() - t0
ids = tokenizer.encode(out, encode_special_tokens = True)
n = int(ids.numel()) if hasattr(ids, "numel") else len(ids)
print(f"decode: {n} tokens in {t_gen:.2f}s -> {n/t_gen:.2f} tok/s")
print("--- response ---")
print(out)
print("--- end ---")
print("RESULT cc =", torch.cuda.get_device_capability(0), "| decode tok/s = %.2f" % (n/t_gen))
