"""
If the student is a ViT
using the following CrossModel and InnerModel Module
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# define basic attention
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.) -> None:
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        # assert dim == heads * dim_head, "dim must be equal to heads * dim_head"
        self.attn = Attention(dim, heads, dim_head, dropout)
        self.ffn = FeedForward(dim, mlp_dim, dropout)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

# class CrossModelBlock(nn.Module):
#     """
#     Cross Model interaction
#     """
#     def __init__(self, dim, num_models=3, heads=8, dropout=0.):
#         super(CrossModelBlock, self).__init__()
#         """
#         dim_list: list of dimensions of each teacher
#         """
#         self.num_models = num_models
#         self.dim = dim * num_models
#         dim_head = self.dim // heads
#         self.block = TransformerBlock(dim=self.dim, heads=heads, dim_head=dim_head, mlp_dim=dim, dropout=dropout)

#     def forward(self, xs):
#         """
#         xs: list of features [b, L, c]
#         """
#         x = torch.cat(xs, dim=-1) # [b, L, d*self.num_models]
#         x = self.block(x)
#         xs = torch.chunk(x, chunks=self.num_models, dim=-1)
#         return xs


class CrossModelBlock(nn.Module):
    """
    Cross Model interaction
    """
    def __init__(self, in_dim, dim, num_models=3, heads=8, dropout=0.):
        super(CrossModelBlock, self).__init__()
        """
        dim_list: list of dimensions of each teacher
        """
        self.num_models = num_models
        dim_head = dim // heads
        self.block = TransformerBlock(dim=in_dim, heads=heads, dim_head=dim_head, mlp_dim=dim*2, dropout=dropout)

    def forward(self, xs):
        """
        xs: list of features [b, L, c]
        """
        x = torch.cat(xs, dim=1) # [b, nL, d*self.num_models]
        x = self.block(x)
        xs = torch.chunk(x, chunks=self.num_models, dim=1)
        return xs

class InnerModelBlock(nn.Module):
    """
    Inner Model interaction
    """
    def __init__(self, in_dim, dim, num_models=3, heads=8, dropout=0.):
        super(InnerModelBlock, self).__init__()
        self.num_models = num_models
        dim_head = dim // heads
        self.block = TransformerBlock(dim=in_dim, heads=heads, dim_head=dim_head, mlp_dim=dim*2, dropout=dropout)

    def forward(self, xs):
        """
        xs: list of features [b, c, h, w]
        """
        b, L, d = xs[0].shape
        xs = [x.unsqueeze(1) for x in xs] # [ [b, 1, L, d] ]
        x = torch.cat(xs, dim=1).view(b*self.num_models, L, d)
        x = self.block(x)
        x = x.view(b, self.num_models, L, d)
        xs = torch.chunk(x, chunks=self.num_models, dim=1)
        xs = [item.squeeze(1) for item in xs]
        return xs

class TokenPredictHead(nn.Module):
    """
    Predict head of tokens
    """
    def __init__(self, dim, out_dim):
        super(TokenPredictHead, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, out_dim)
    
    def forward(self, x):
        x = self.norm(x)
        if x.ndim == 3:
            x = x.mean(dim=1)  # [b, l, d] -> [b, d]
        x = self.proj(x)
        return x

class PatchPredictHead(nn.Module):
    """
    Predict patch 
    """
    def __init__(self, dim, out_dim):
        super(PatchPredictHead, self).__init__()
        self.layer = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim)
        )
    
    def forward(self, x):
        return self.layer(x)