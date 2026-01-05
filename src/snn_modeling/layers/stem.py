import torch.nn as nn
import torch
import snntorch as snn
from .residual_blocks import ConvSpiking
from .neurons import TimeDistributed, SwiGLU
import torch.nn.functional as F

class StemLayer(nn.Module):
    def __init__(self, in_channels):
        super(StemLayer, self).__init__()
        
        self.layer = ConvSpiking(
            in_channels,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            spike_model=nn.Identity,
            use_norm=False)
        
        self.norm = TimeDistributed(nn.InstanceNorm2d(64, affine=True,eps=1e-6))
        self.act = nn.SiLU()
    def forward(self, x):

        return self.act(self.norm(self.layer(x)))
    
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
    def __init__(self, feature_dim=512, head_dim=128):
        super(ProjectionHead, self).__init__()
        self.supcon_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(feature_dim, head_dim, kernel_size=1)
        )
    def forward(self, features):
        T, B, C, H, W = features.shape
        proj = self.supcon_head(features.view(T * B, C, H, W)) + 1e-6 
        embedding = F.normalize(proj.view(T * B, -1), dim=1)
        return embedding
    
class TemporalViTBlock(nn.Module):
    def __init__(self, in_channels,
                 num_heads=8,
                 p_drop=0.1):
        super(TemporalViTBlock, self).__init__()
        self.in_channels = in_channels
        self.norm1 = nn.LayerNorm(in_channels)
        self.attn = nn.MultiheadAttention(embed_dim=in_channels,
                                          num_heads=num_heads,
                                          dropout=p_drop)
        self.norm2 = nn.LayerNorm(in_channels)
        self.mlp = SwiGLU(in_channels, p_drop=p_drop)
        self.dropout = nn.Dropout(p_drop)
        
    def forward(self, x):
        T, B, C, H, W = x.shape
        x_flat = x.mean(dim=[3,4])  # Average pool over spatial dimensions
        src = self.norm1(x_flat)
        attn_output, _ = self.attn(src, src, src)
        x_flat = x_flat + self.dropout(attn_output)
        src = self.norm2(x_flat)
        mlp_output = self.mlp(src)
        x_flat = x_flat + mlp_output
        context = x_flat.view(T, B, C, 1, 1)
        return x + context
    
class TemporalGCBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(TemporalGCBlock, self).__init__()
        self.conv_mask = nn.Conv1d(in_channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.transform = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, kernel_size=1),
            nn.LayerNorm([in_channels // reduction, 1]),
            nn.SiLU(inplace=True),
            nn.Conv1d(in_channels // reduction, in_channels, kernel_size=1)
        )
    
    def forward(self, x):
        T, B, C, H, W = x.shape
        x_flat = x.mean(dim=[3,4])  # Average pool over spatial dimensions
        x_ctx = x_flat.permute(1, 2, 0) # B x C x T
        mask = self.conv_mask(x_ctx)
        attn = self.softmax(mask)
        context = torch.matmul(x_ctx, attn.permute(0, 2, 1))
        context = self.transform(context).permute(2,0,1).view(1,B,C,1,1)
        return x + context
