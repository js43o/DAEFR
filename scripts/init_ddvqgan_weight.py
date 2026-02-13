import torch

ckpt_state = torch.load('experiments/associate_2.ckpt')['state_dict']
new_ckpt = {}
new_ckpt['state_dict'] = {}

for name, tensor in ckpt_state.items():
  if name.startswith('vqvae.decoder.'):
    new_ckpt['state_dict']['decoder_r.' + name.split('vqvae.decoder.')[1]] = tensor
    new_ckpt['state_dict']['decoder_f.' + name.split('vqvae.decoder.')[1]] = tensor
  elif name.startswith('vqvae.'):
    new_ckpt['state_dict'][name.split('vqvae.')[1]] = tensor

torch.save(new_ckpt, 'pretrained/ddvqgan.ckpt')
