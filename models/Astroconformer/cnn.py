import torch
import torch.nn as nn

class ConvBlock(nn.Module):
  def __init__(self, encoder_dim: int, kernel_size: int = 3, **kwargs) -> None:
    super().__init__()
    self.layers = nn.Sequential(
        nn.Conv1d(in_channels=encoder_dim,
                out_channels=encoder_dim,
                kernel_size=kernel_size,
                stride=1, padding='same', bias=False),
        nn.BatchNorm1d(num_features=encoder_dim),
        nn.SiLU(),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:  
    x = x.transpose(1, 2)
    return self.layers(x).transpose(1, 2)