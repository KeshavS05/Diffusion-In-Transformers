import torch
import torch.nn as nn


device = "cuda" if torch.cuda.is_available() else "cpu"

class ViTBlock_InContext(nn.Module):
   def __init__(self, d):
      super().__init__()
      self.ln1 = nn.LayerNorm(d)
      self.SelfAttention = nn.MultiheadAttention(d, 8, batch_first=True)
      self.ln2 = nn.LayerNorm(d)
      self.mlp = nn.Sequential(nn.Linear(d, 3 * d),
                               nn.GELU(),
                               nn.Linear(3 * d, d))
      
   def forward(self, x, cond=None):                                           
      normalized_x = self.ln1(x)                                                 # Apply first LayerNorm
      x = x + self.SelfAttention(normalized_x, normalized_x, normalized_x)[0]    # Q = K = V => Self attention
      normalized_post_attn = self.ln2(x)                                         # Apply second layer norm
      x = x + self.mlp(normalized_post_attn)                                     # Pointwise Feedforward MLP 
      return x 
   