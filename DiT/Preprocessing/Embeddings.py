import torch
import torch.nn as nn



device = "cuda" if torch.cuda.is_available() else "cpu"

def positional_embedding(T, d):
   n = torch.tensor(10000).to(device=device, dtype=torch.float32)
   positional_embeddings = torch.zeros((T, d)).to(device=device, dtype=torch.float32)
   
   positions = torch.arange(end=T, device=device, dtype=torch.float32)                             # i for i in range(T)
   div = n**((2/d) * torch.arange(start=0, end=d, step=2, device=device, dtype=torch.float32))     # n**(2i/d) where d is cut in half (half for sin, half for cos)
   
   positional_embeddings[:, 0::2] = torch.sin(positions.unsqueeze(1) / div.unsqueeze(0))           # All even positions are sin
   positional_embeddings[:, 1::2] = torch.cos(positions.unsqueeze(1) / div.unsqueeze(0))           # All odd positions are cos
      
   return positional_embeddings




def timestep_embedding(timesteps, d):

   n = torch.tensor(10000, device=device, dtype=torch.float32)   
   freqs = torch.exp(-1 * torch.log(n) * torch.arange(0, d//2, device=device, dtype=torch.float32) / (d//2))
   timesteps = timesteps.reshape(-1, 1)
   freqs = freqs.reshape(1, -1)
   mult = timesteps * freqs
   
   timestep_embeddings = torch.cat((torch.sin(mult), torch.cos(mult)), -1)
   return timestep_embeddings






class InContextConditioning(nn.Module):
   def __init__(self, d, t_dim, y_dim):
      super().__init__()
      self.t_dim = t_dim
      self.transform_t = nn.Sequential(nn.Linear(t_dim, d),                # Transform timestep embedding from t_dim to d-dimensional embedding
                                       nn.SiLU(),
                                       nn.Linear(d, d))
      
      self.y_lookup = nn.Embedding(y_dim, d)                               # Lookup table to map from y class label to d-dimensional embedding
      
   def forward(self, t, y):
      time_embed = self.transform_t(timestep_embedding(t, self.t_dim))     # B x d
      y_embed = self.y_lookup(y)                                           # B x d
      return torch.stack((time_embed, y_embed), dim=1)                     # B x 2 x d