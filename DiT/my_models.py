import torch
import torch.nn as nn

# from Preprocessing import VAE, Patchify, Embeddings
from Preprocessing import Patchify, Embeddings
from ConditionMethods import InContext, CrossAttn, AdaLN_Zero



device = "cuda" if torch.cuda.is_available() else "cpu"

class DiTBlockList(nn.Module):
   def __init__(self, d, blocks, type: str):
      super().__init__()
      self.type_ = type
      if (type == "In_Context"):
         self.blocks = nn.ModuleList([InContext.ViTBlock_InContext(d) for _ in range(blocks)])
      elif (type == "Cross_Attn"):
         self.blocks = nn.ModuleList([CrossAttn.ViTBlock_CrossAttn(d) for _ in range(blocks)])
      else:
         self.blocks = nn.ModuleList([AdaLN_Zero.ViTBlock_AdaLN_Zero(d) for _ in range(blocks)])
      
   def forward(self, x, cond):
      if (self.type_ == "In_Context"):
         x = torch.cat([x, cond], dim=1)     # Add conditioned embeddings directly to token embeddings to get x = B x T+2 x d
      for block in self.blocks:
         x = block(x, cond)
      
      return x


class DiT(nn.Module):
   
   # End-to-end Diffusion Transformer

   # Notes:
   # Assumes input of 256 x 256 x 3 images which are mapped to 32 x 32 x 4 latent space images (p.4 "Patchify")
   # Output from Unpatchify is [B, 8, H, W] where first 4 are noise, second 4 are sigma (variance params) (p.5 "Transformer Decoder")

   # Args:
   # p: patch size (2, 4, 8 used in paper (p.4 "Patchify"))
   # d: token dimension (p.4 "Figure 4")
   # blocks: number of stacked ViT blocks
   # t_dim: timestep embedding dim
   # y_dim: number of classes for label embedding
   # in_ch: input channels
   # type: Type of conditioning method to use (In_Context, Cross_Attn, AdaLN-Zero). Default is AdaLN-Zero
   
   def __init__(self, p, d=512, blocks=12, t_dim=256, y_dim=1000, in_ch= 4, type=""):
      super().__init__()
      self.p = p
      self.d = d
      self.in_ch = in_ch
      self.type_ = type

      self.patchify = Patchify.Patchify(p=p, d=d)
      self.in_context = Embeddings.InContextConditioning(d=d, t_dim=t_dim, y_dim=y_dim)
      self.blocks = DiTBlockList(d=d, blocks=blocks, type=type)
      self.unpatchify = Patchify.Unpatchify(d=d, p=p)

   def forward(self, x, t, y):
      
      # Args
      # x = B x C x I x I (In the DiT paper I is normally 32 and C is usually 4)
      # t = timesteps = B
      # y = class labels = B

      # Apply patchify to transform latent from B x 4 x 32 x 32 to B x T x d
      tokens = self.patchify(x).to(device=device, dtype=torch.float32)
      _, T, d = tokens.shape

      # Add positional embeddings (T x d) to each token
      pos = Embeddings.positional_embedding(T, d).to(tokens.device, tokens.dtype)
      tokens = tokens + pos.unsqueeze(0)     # B x T x d + 1 x T x d = B x T x d
      
      # Build conditioning tokens (B, 2, d)
      cond_tokens = self.in_context(t, y)
      cond_tokens = cond_tokens.to(tokens.dtype).to(tokens.device)

      # Run all stacked DiT blocks using {type} conditioning
      seq = self.blocks(tokens, cond_tokens)                                            # [B, T+2, d]

      # If we used in-context conditioning, remove the embedded conditioned tokens (i.e transform from B x T+2 x d to B x T x d)
      if self.type_ == "In_Context":
         seq = seq[:, :T, :]                                        # [B, T, d]

      # Unpatchify back to latent space, returning a B x 2C x 32 x 32 tensor where the first B x C is for the noise and the second is for the variance
      out = self.unpatchify(seq)
      return out