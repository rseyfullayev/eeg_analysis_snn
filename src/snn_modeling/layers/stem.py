import torch.nn as nn
import torch
import snntorch as snn
from .residual_blocks import ConvSpiking
from .neurons import TimeDistributed

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