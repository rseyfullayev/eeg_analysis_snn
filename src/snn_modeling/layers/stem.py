import torch.nn as nn
import snntorch as snn
from .residual_blocks import ConvBnSpiking

class StemLayer(nn.Module):
    def __init__(self, in_channels):
        super(StemLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
    def forward(self, x):
        return self.layer(x)
    
class ClassifierHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super(ClassifierHead, self).__init__()
        self.head = nn.Conv2d(in_features, num_classes, kernel_size=1)
    def forward(self, x):
        return self.head(x)
    
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, p_drop=0.2, spike_model=snn.Leaky, **neuron_params):
        super(BottleneckBlock, self).__init__()
        self.bottleneck = nn.Sequential(
            ConvBnSpiking(in_channels, in_channels//2, kernel_size=3, padding=1, spike_model=spike_model, **neuron_params),
            nn.Dropout2d(p=p_drop),
            ConvBnSpiking(in_channels//2, in_channels, kernel_size=3, padding=1, spike_model=spike_model, **neuron_params),
        )
    def forward(self, x):
        return self.bottleneck(x)