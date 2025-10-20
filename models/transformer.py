import torch
import math

import torch.nn as nn
import torch.nn.functional as F

activations = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "elu": nn.ELU(),
    "leaky_relu": nn.LeakyReLU(),
    "gelu": nn.GELU(),
    "tanh": nn.Tanh(),
    "silu": nn.SiLU()
}

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.0, activation='relu', output_activation=None, input_activation=None):
        super().__init__()
        layers = []
        if input_activation is not None:
            layers.append(input_activation())
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activations[activation])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        if output_activation is not None:
            layers.append(activations[output_activation])
        self.network = nn.Sequential(*layers)
            
    def forward(self, x):
        return self.network(x)
    
    def forward_ll(self, x):
        out = self.network(x)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        attn_mask = mask
        
        # Flash attention (scaled dot-product attention)
        attn_output = F.scaled_dot_product_attention(
            Q, K, V, 
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False
        )
        
        # Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.w_o(attn_output)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, activation=nn.GELU()):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, factor_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, factor_ff*d_model, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class Transformer(nn.Module):
    def __init__(self, input_dim,model_dim,decoder_hidden_dims,output_dim,nlayers=4,nhead=8,
                 fan_factor=4,att_drop=0.1,att_activation='gelu',decoder_activation='gelu',decoder_drop=0.1, **kwargs):
        super().__init__()
        self.embedding = nn.Linear(input_dim, model_dim)        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(model_dim, nhead, fan_factor, dropout=att_drop)
            for _ in range(nlayers)
        ])
        
        self.dropout = nn.Dropout(att_drop)
        self.decoder = MLP(input_dim=model_dim,hidden_dims=decoder_hidden_dims,output_dim=output_dim,
                           activation=decoder_activation,dropout=decoder_drop)
        self.nhead = nhead
        print(f"Transformer initialized with input_dim={input_dim}, model_dim={model_dim}, output_dim={output_dim}, nlayers={nlayers}, nhead={nhead}, fan_factor={fan_factor}, att_drop={att_drop}, decoder_hidden_dims={decoder_hidden_dims}")
        
    def forward(self, x, mask=None):
        # x: (B, features, n_particles)
        # mask: (B, 1, n_particles) -- real particle = 1, padded = 0
        x = x.transpose(1, 2) # (B, n_particles, features)
        n_particles = x.shape[1]
        x = self.embedding(x)
        x = self.dropout(x)
        if torch.isnan(x).any():
            raise ValueError("NaN detected in input to transformer")

        # make mask suitable for transformer, shape (B, n_particles, n_particles)
        # where True = take part in attention
        if mask is None:
            attn_mask = mask
        else:
            attn_mask = mask.bool().repeat_interleave(n_particles,dim=1) # (B, n_particles, n_particles)
            attn_mask = attn_mask.unsqueeze(1) # (B, 1, n_particles, n_particles) to broadcast over heads
        
        # Pass through transformer blocks
        for i,block in enumerate(self.transformer_blocks):
            x = block(x, attn_mask)
            if torch.isnan(x).any():
                raise ValueError(f"NaN detected in output of transformer block {i}")

        if mask is not None:
            x = x * mask.transpose(1,2) # reshape mask to (B, n_particles, 1)
        x = x.mean(dim=1) # average over sequence length (particles)
        x = self.decoder(x)
        if torch.isnan(x).any():
            raise ValueError("NaN detected in output of transformer")
            
        return x