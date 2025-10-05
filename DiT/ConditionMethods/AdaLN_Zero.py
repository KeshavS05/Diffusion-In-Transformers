import torch
import torch.nn as nn



device = "cuda" if torch.cuda.is_available() else "cpu"

class ViTBlock_AdaLN_Zero(nn.Module):
   def __init__(self, d):
      super().__init__()
      self.ln1 = nn.LayerNorm(d, elementwise_affine=False)                             # First layer norm
      self.ln2 = nn.LayerNorm(d, elementwise_affine=False)                             # Second layer norm
      
      self.SelfAttention = nn.MultiheadAttention(d, 8, batch_first=True)               # Self-Attention block
      self.get_params = nn.Sequential(nn.SiLU(),
                                      nn.Linear(d, 6 * d))                             # Return 6 vectors each of d length (y1, b1, a1, y2, b2, a2)
   
      self.mlp = nn.Sequential(nn.Linear(d, 3 * d),
                               nn.GELU(),
                               nn.Linear(3 * d, d))
      
      nn.init.zeros_(self.get_params[1].weight)                                        # Use identity of self attention to start (no scale or shift)
      nn.init.zeros_(self.get_params[1].bias)
      
   def forward(self, x, cond):
      cond = cond.sum(dim=1)                                                           # Transform conditioned embeddings from B x 2 x d to B x d (sum both embeddings)
      
      normalized_x = self.ln1(x)                                                       # First layer norm
      params = (self.get_params(cond)).unsqueeze(1)                        
      y1, b1, a1, y2, b2, a2 = torch.chunk(params, chunks=6, dim=-1)                   # Split params into six vector parameters
      normalized_x = (1 + y1) * normalized_x + b1                                      # Apply first scale and shift
      x = x + a1 * self.SelfAttention(normalized_x, normalized_x, normalized_x)[0]     # Q = K = V => Self attention, apply scale
      
      normalized_post_attn = self.ln2(x)                                               # Process repeated once more before returning
      normalized_post_attn = (1 + y2) * normalized_post_attn + b2
      x = x + a2 * self.mlp(normalized_post_attn)
      return x