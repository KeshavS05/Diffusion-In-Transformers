import torch
import torch.nn as nn


device = "cuda" if torch.cuda.is_available() else "cpu"
   
# Class to patchify the latent image (i.e 4 x 32 x 32 to T x d)
class Patchify(nn.Module):
   def __init__(self, p, d, in_ch=4):
      super().__init__()
      self.p = p   # Stride/Kernel Size
      self.d = d   # Embedding Dimensionm
      self.patchify = nn.Conv2d(in_ch, d, p, p)
   
   def forward(self, z):
      patch = self.patchify(z)                                    # Apply patchify to get B x d x 32/p x 32/p
      patch = patch.flatten(start_dim = 2, end_dim = 3)           # Flatten to get B x d x (32/p x 32/p)
      patch = patch.transpose(1, 2)                               # Transpose into B x T x d
      return patch
   

class Unpatchify(nn.Module):
   def __init__(self, d, p, in_ch=4):
      super().__init__()
      self.p = p
      self.in_ch = in_ch
      self.norm = nn.LayerNorm(d)
      self.proj = nn.Linear(d, p * p * 2 * in_ch)

   def forward(self, tokens: torch.Tensor) -> torch.Tensor:
      # tokens: [B, T, D], with T = (I/p)^2 and I=32 here
      B, T, d = tokens.shape
      g = int(T ** 0.5)                            # g = sqrt(T) = I/p
      I = g * self.p                               # I = g * p

      x = self.norm(tokens)                        # Apply layer norm
      x = self.proj(x)                             # B x T x d to B x T x (p * p * 2 * C)
      x = x.reshape(B, g, g, self.p, self.p, 8)    # T = (I/p)^2 so B x T x (p * p * 2 * C) => B x (g * g) x (p * p * 2 * C) => B x g x g x p x p x (2 * C)
      x = x.permute(0, 5, 1, 3, 2, 4)              # B x g x g x p x p x (2 * C) => B x (2 * C) x g x p x g x p => B x (2 * C) x (I/p) x p x (I/p) x p => B x (2 * C) x I x I
      x = x.reshape(B, 8, I, I)                    # [B, 2C, 32, 32]
      return x