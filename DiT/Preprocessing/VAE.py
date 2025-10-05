import torch
import torch.nn as nn
from diffusers import AutoencoderKL

device = "cuda" if torch.cuda.is_available() else "cpu"
scale = 0.18215

class VAE(nn.Module):
   def __init__(self, arg):   
      super().__init__()
      self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{arg}")       # arg = mse or ema
      self.eval()       # Frozen encoder, don't train so set to eval mode
      self.vae.to(device)
      
   def encode(self, x):
      # X has a size of Batch * Channel * Height * Width
      encoded = self.vae.encode(x).latent_dist   # encoded outputs of encoder in form of mean and logvar of gaussian dist
      z = encoded.sample() * scale      # B x 4 x 32 x 32
      print(z.shape)
      return z
   
   def decode(self, z):
      to_return = self.vae.decode(z / scale).sample
      print(to_return.shape)
      return to_return