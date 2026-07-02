import torch.nn as nn
import torch
import snntorch as snn
from .residual_blocks import ConvSpiking
from .neurons import TimeDistributed, SwiGLU
import torch.nn.functional as F
import math
    
class ClassifierHead(nn.Module):
    def __init__(self, in_features, num_classes, kernel_size=1):
        super(ClassifierHead, self).__init__()

        self.head = TimeDistributed(nn.Conv2d(in_features, num_classes, kernel_size=kernel_size, bias=True))
    
    def forward(self, x):
        return self.head(x)
        
    
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, p_drop=0.2, spike_model=snn.Leaky, **neuron_params):
        super(BottleneckBlock, self).__init__()
        self.conv1 = ConvSpiking(in_channels, in_channels // 2, kernel_size=3, padding=1, spike_model=spike_model, **neuron_params)
        self.drop = TimeDistributed(nn.Dropout2d(p=p_drop))
        self.conv2 = ConvSpiking(in_channels // 2, in_channels, kernel_size=3, padding=1, spike_model=spike_model, **neuron_params)

    def forward(self, x):
        x = self.conv1(x)
        x = self.drop(x)
        x = self.conv2(x)
        return x

class ProjectionHead(nn.Module):
    def __init__(self, feature_dim=512, head_dim=128, use_batchnorm=False):
        super(ProjectionHead, self).__init__()

        self.supcon_head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim) if use_batchnorm else nn.GroupNorm(1, feature_dim, affine=True),
            nn.SiLU(inplace=False),
            nn.Conv2d(feature_dim, head_dim, kernel_size=1)
        )
    def forward(self, features):
        B, D = features.shape
        features = features.reshape(B, D, 1, 1)  # Reshape to (B, D, 1, 1) for Conv2d
        proj = self.supcon_head(features)
        embedding = F.normalize(proj.view(B, -1), dim=1, eps=1e-6)
        return embedding

class VIBLayer(nn.Module):
    def __init__(self, in_channels):
        super(VIBLayer, self).__init__()
        self.fc_mu = nn.Linear(in_channels, in_channels//8)
        self.fc_logvar = nn.Linear(in_channels, in_channels//8)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding for spatial dimensions.
    Input shape: Batch, Trial, Windows, Feature Dim
    Applying encoding across windows per trial (for each trial, encoding should be reset)
    """
    def __init__(self, in_channels, d_model, max_windows):
        super(PositionalEncoding, self).__init__()
        base = math.ceil(max_windows/(2*math.pi))
        pe = torch.zeros(1, max_windows, d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(base) / d_model))
        position = torch.arange(max_windows).unsqueeze(1)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    
    def forward(self, x):
        # x shape: [Batch, Windows, Channels]
        num_windows = x.size(1)
        return x + self.pe[:, :num_windows, :]


class WindowReRanker(nn.Module):
    def __init__(self, in_channels, feature_dim=256, max_windows=400):
        super(WindowReRanker, self).__init__()
        self.pos_enc = PositionalEncoding(in_channels, feature_dim, max_windows)
        self.scorer = SwiGLU(feature_dim, feature_dim//8, 1, p_drop=0.1)
    
    def forward(self, x, mask=None):
        x = self.pos_enc(x)
        scores = self.scorer(x)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        return F.softmax(scores, dim=2)
        
    
class TemporalViTBlock(nn.Module):
    def __init__(self, in_channels,
                 num_heads=8,
                 p_drop=0.1,
                 use_batchnorm=False):
        super(TemporalViTBlock, self).__init__()
        self.in_channels = in_channels
        self.use_batchnorm = use_batchnorm
        self.norm1 = nn.BatchNorm1d(in_channels) if use_batchnorm else nn.LayerNorm(in_channels)
        self.attn = nn.MultiheadAttention(embed_dim=in_channels,
                                          num_heads=num_heads,
                                          dropout=p_drop)
        self.norm2 = nn.BatchNorm1d(in_channels) if use_batchnorm else nn.LayerNorm(in_channels)
        self.mlp = SwiGLU(in_channels, p_drop=p_drop)
        self.dropout = nn.Dropout(p_drop)
        
    def forward(self, x):
        T, B, C, H, W = x.shape
        x_flat = x.mean(dim=[3,4])  # Average pool over spatial dimensions
        
        if self.use_batchnorm:
            src = x_flat.permute(1, 2, 0) # [B, C, T] for BatchNorm1d
            src = self.norm1(src).permute(2, 0, 1) # back to [T, B, C]
        else:
            src = self.norm1(x_flat)
            
        attn_output, _ = self.attn(src, src, src)
        x_flat = x_flat + self.dropout(attn_output)
        
        if self.use_batchnorm:
            src = x_flat.permute(1, 2, 0)
            src = self.norm2(src).permute(2, 0, 1)
        else:
            src = self.norm2(x_flat)
            
        mlp_output = self.mlp(src)
        x_flat = x_flat + mlp_output
        context = x_flat.view(T, B, C, 1, 1)
        return x + context
    
class TemporalGCBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, use_batchnorm=False):
        super(TemporalGCBlock, self).__init__()
        self.conv_mask = nn.Conv1d(in_channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.transform = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, kernel_size=1),
            nn.BatchNorm1d(in_channels // reduction) if use_batchnorm else nn.LayerNorm([in_channels // reduction, 1]),
            nn.SiLU(inplace=False),
            nn.Conv1d(in_channels // reduction, in_channels, kernel_size=1)
        )
    
    def forward(self, x):
        T, B, C, H, W = x.shape
        x_flat = x.mean(dim=[3,4])  # Average pool over spatial dimensions
        x_ctx = x_flat.permute(1, 2, 0) # B x C x T
        mask = self.conv_mask(x_ctx)
        attn = self.softmax(mask)
        context = torch.matmul(x_ctx, attn.permute(0, 2, 1))
        context = self.transform(context).permute(2,0,1).view(B,C)
        return context
