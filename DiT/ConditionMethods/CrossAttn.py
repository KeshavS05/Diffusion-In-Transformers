import torch
import torch.nn as nn




device = "cuda" if torch.cuda.is_available() else "cpu"
   
class ViTBlock_CrossAttn(nn.Module):
   def __init__(self, d):
      super().__init__()
      self.ln1 = nn.LayerNorm(d)                                                             # First layer norm
      self.SelfAttention = nn.MultiheadAttention(d, 8, batch_first=True)                     # Self attention block
      
      self.ln2 = nn.LayerNorm(d)                                                             # Second Layer norm
      self.CrossAttention = nn.MultiheadAttention(d, 8, batch_first=True)                    # Cross attention block
      
      self.ln3 = nn.LayerNorm(d)                                                             # Third layer norm
      self.mlp = nn.Sequential(nn.Linear(d, 3 * d),
                               nn.GELU(),
                               nn.Linear(3 * d, d))                                          # Pointwise Feedforward
      
   def forward(self, x, cond_tokens):
      normalized_x = self.ln1(x)                                                             # Apply first layer norm
      x = x + self.SelfAttention(normalized_x, normalized_x, normalized_x)[0]                # Q = K = V => Self attention

      normalized_post_self_attn = self.ln2(x)                                                # Apply second layer norm
      x = x + self.CrossAttention(normalized_post_self_attn, cond_tokens, cond_tokens)[0]    # Cross attention with conditioned tokens
      
      normalized_post_cross_attn = self.ln3(x)                                               # Apply third layer norm
      x = x + self.mlp(normalized_post_cross_attn)                                           # Output after MLP

      return x