from llama.model import ModelArgs
from llama.tokenizer import Tokenizer
from llama.generation import Llama
import os
from model import Transformer
import torch
import json
from pathlib import Path

model_dir = 'llama-2-7b'
checkpoints = sorted(Path(model_dir).glob("*.pth"))
ckpt_path = checkpoints[-1]
checkpoint = torch.load(ckpt_path, map_location='cpu')
del checkpoint['rope.freqs']
tokenizer_model = 'tokenizer.model'
param_json = os.path.join(model_dir, 'params.json')
with open(param_json, 'r') as f:
    params = json.load(f)

args = ModelArgs(
    max_seq_len = 1024,
    max_batch_size = 1,
    **params)

tokenizer = Tokenizer(tokenizer_model)
args.vocab_size = tokenizer.n_words
model = Transformer(args)

torch.manual_seed(1)
torch.set_default_tensor_type(torch.HalfTensor)
print(f"Loading checkpoint from {ckpt_path}")
model.load_state_dict(checkpoint)
print("Done!")

llama = Llama(model, tokenizer)

text_completion = llama.text_completion(
    prompts=["what's the capital of France?"],
    device='cpu',
)

print(text_completion)