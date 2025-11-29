import torch.nn as nn
import snntorch as snn
from .residual_blocks import ConvSpiking, ConvBnSpiking

class StemLayer(nn.Module):
    def __init__(self, in_channels):
        super(StemLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.act = nn.LeakyReLU(inplace=True) 
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        if x.dim() == 5:
            T, B, C, H, W = x.shape
    
            x_flat = x.reshape(T * B, C, H, W)

            out = self.conv(x_flat)
            out = self.bn(out)
            out = self.act(out)
            out = self.pool(out)
            
            _, C_out, H_out, W_out = out.shape

            out = out.reshape(T, B, C_out, H_out, W_out)
            return out
        else:
            x = self.conv(x)
            x = self.bn(x)
            x = self.act(x)
            x = self.pool(x)
            return x
    
class ClassifierHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super(ClassifierHead, self).__init__()
        self.head = nn.Conv2d(in_features, num_classes, kernel_size=1, bias=False)
    def forward(self, x):
        return self.head(x)
    
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, p_drop=0.2, spike_model=snn.Leaky, **neuron_params):
        super(BottleneckBlock, self).__init__()
        self.conv1 = ConvSpiking(in_channels, in_channels // 2, kernel_size=3, padding=1, spike_model=spike_model, **neuron_params)
        self.drop = nn.Dropout2d(p=p_drop)
        self.conv2 = ConvSpiking(in_channels // 2, in_channels, kernel_size=3, padding=1, spike_model=spike_model, **neuron_params)

    def forward(self, x):
        x = self.conv1(x)
        
        if x.dim() == 5:
            T, B, C, H, W = x.shape
            x = x.reshape(T * B, C, H, W) 
            x = self.drop(x)            
            x = x.reshape(T, B, C, H, W)  
        else:
            x = self.drop(x)
        
        x = self.conv2(x)
        return x