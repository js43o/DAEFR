import torch
from safetensors.torch import load_file
"""
ckpt = load_file('experiments/02/checkpoints/epoch_0010/model_1.safetensors')

print(ckpt.keys())

with open('_dual_vqgan_keys3.txt', 'w') as f:
  for key in ckpt.keys():
    f.write(key + '\n')
"""

ckpt = torch.load('experiments/ddvqgan.ckpt')['state_dict']

with open('_ddvqgan_keys.txt', 'w') as f:
  for key in ckpt.keys():
    f.write(key + '\n')