import torch
import torch.nn as nn
from torch import Tensor

from .conformer import ConformerEncoder
from .mhsa_pro import RotaryEmbedding
from .ResNet18 import ResNet18

class Astroconformer(nn.Module):
  def __init__(self, in_channels=2, encoder_dim=128, extractor_kernel_size=5, conv_kernel_size=3, output_dim=8,
               encoder=["mhsa_pro", "conv", "conv"], norm="postnorm", num_layers=5, num_heads=8, 
               dropout_p=0.1, timeshift=False, **kwargs) -> None:
    super(Astroconformer, self).__init__()
    self.head_size = encoder_dim // num_heads
    self.rotary_ndims = int(self.head_size * 0.5)
    
    self.extractor = nn.Sequential(nn.Conv1d(in_channels = in_channels,
            kernel_size = extractor_kernel_size, out_channels = encoder_dim, stride = extractor_kernel_size, padding = 0, bias = True),
                    nn.BatchNorm1d(encoder_dim),
                    nn.SiLU(),
    )
    
    self.pe = RotaryEmbedding(self.rotary_ndims)
    
    self.encoder = ConformerEncoder(
        encoder=encoder,
        num_layers=num_layers,
        norm=norm,
        num_heads=num_heads,
        encoder_dim=encoder_dim,
        dropout_p=dropout_p,
        kernel_size=conv_kernel_size,
        timeshift=timeshift,
        **kwargs
    )
    
    self.pred_layer = nn.Sequential(
        nn.Linear(encoder_dim, encoder_dim),
        nn.SiLU(),
        nn.Dropout(p=dropout_p),
        nn.Linear(encoder_dim,output_dim),
    )
    
  def forward(self, inputs: Tensor) -> Tensor:
    x = inputs #initial input_size: [B, d_in, L]
    x = self.extractor(x) # x: [B, encoder_dim, L]
    x = x.permute(0,2,1) # x: [B, L, encoder_dim]
    RoPE = self.pe(x, x.shape[1]) # RoPE: [2, B, L, encoder_dim], 2: sin, cos
    x = self.encoder(x, RoPE) # x: [B, L, encoder_dim]
    x = x.mean(dim=1) # x: [B, encoder_dim]
    x = self.pred_layer(x) # x: [B, d_out]
    return x
    
model_dict = {
          'Astroconformer': Astroconformer,
          'ResNet18': ResNet18,
      }
