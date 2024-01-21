
import os
import json
import torch
from llama import ModelArgs
from model import Transformer
import exports

llama_ckpt_dir = 'llama-2-7b'
llama_ckpt_path = os.path.join(llama_ckpt_dir, 'consolidated.00.pth')
model_args_path = os.path.join(llama_ckpt_dir, 'params.json')
max_seq_len = 1024
max_batch_size = 1
llama_torchsharp_weights_path = os.path.join(llama_ckpt_dir, 'llama-2-7b.pt')

checkpoint = torch.load(llama_ckpt_path, map_location="cpu")
with open(model_args_path, "r") as f:
    params = json.loads(f.read())

model_args: ModelArgs = ModelArgs(
    max_seq_len=max_seq_len,
    max_batch_size=max_batch_size,
    **params,
    )

model_args.vocab_size = 32000
# use bf16 as default
torch.set_default_dtype(torch.bfloat16)
model = Transformer(model_args)
model.load_state_dict(checkpoint, strict=False)
model.eval()
print(model.state_dict().keys())
with open(llama_torchsharp_weights_path, 'wb') as f:
    exports.save_state_dict(model.state_dict(keep_vars=False), f)