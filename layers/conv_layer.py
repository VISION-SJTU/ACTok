"""
If the student is a ConvNet
using the following CrossModel and InnerModel Module
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class CrossModelBlock(nn.Module):
    """
    Cross Model interaction
    """
    def __init__(self, dim, num_models=1, groups=1):
        super(CrossModelBlock, self).__init__()
        self.num_models = num_models
        self.norm = LayerNorm(normalized_shape=dim*num_models, data_format="channels_first")
        self.layer = nn.Sequential(
            nn.Conv2d(num_models*dim, num_models*dim, kernel_size=3, padding=1, groups=num_models*dim),
            nn.GELU(),
            nn.Conv2d(num_models*dim, num_models*dim, kernel_size=1, padding=0, groups=groups)
        )

    def forward(self, xs):
        """
        xs: list of features [b, c, h, w]
        """
        b, c, h, w = xs[0].shape
        xs = [x.unsqueeze(2) for x in xs]
        x = torch.cat(xs, dim=2).view(b, c*self.num_models, h, w)
        identity = x
        x = self.norm(x)
        x = self.layer(x)
        x = x + identity
        x = x.view(b, c, self.num_models, h, w)
        xs = torch.chunk(x, chunks=self.num_models, dim=2)
        xs = [item.squeeze(2) for item in xs]
        return xs

class InnerModelBlock(nn.Module):
    """
    Inner Model interaction
    """
    def __init__(self, dim, num_models=1, groups=1):
        super(InnerModelBlock, self).__init__()
        self.num_models = num_models
        self.norm = LayerNorm(normalized_shape=dim, data_format="channels_first")
        self.layer = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, groups=groups)
        )

    def forward(self, xs):
        """
        xs: list of features [b, c, h, w]
        """
        b, c, h, w = xs[0].shape
        xs = [x.unsqueeze(1) for x in xs] # [ [b, 1, c, h, w] ]
        x = torch.cat(xs, dim=1).view(b*self.num_models, c, h, w)
        identity = x
        x = self.norm(x)
        x = self.layer(x)
        x = x + identity
        x = x.view(b, self.num_models, c, h, w)
        xs = torch.chunk(x, chunks=self.num_models, dim=1)
        xs = [item.squeeze(1) for item in xs]
        return xs

class TokenPredictHead(nn.Module):
    """
    Predict head of tokens
    """
    def __init__(self, dim, out_dim):
        super(TokenPredictHead, self).__init__()
        self.norm = LayerNorm(dim, data_format="channels_first")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(dim, out_dim)
    
    def forward(self, x):
        x = self.norm(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.proj(x)
        return x

class PatchPredictHead(nn.Module):
    """
    Predict patch 
    """
    def __init__(self, dim, out_dim):
        super(PatchPredictHead, self).__init__()
        self.layer = nn.Sequential(
            LayerNorm(dim, data_format="channels_first"),
            nn.Conv2d(dim, out_dim, kernel_size=1, padding=0)
        )
    
    def forward(self, x):
        return self.layer(x)