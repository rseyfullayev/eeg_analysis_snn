import torch.nn as nn
import snntorch as snn

class ConvBnSpiking(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, spike_model=snn.Leaky, **neuron_params):
        super(ConvBnSpiking, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.spike = spike_model(**neuron_params)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.spike(x)
        return x
    
class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    
class SpikingResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, spike_model=snn.Leaky, **neuron_params):
        super(SpikingResidualBlock, self).__init__()
        self.conv1 = ConvBnSpiking(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, spike_model=spike_model, **neuron_params)
        self.conv2 = ConvBn(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if stride != 1 or in_channels != out_channels:
            self.downsample = ConvBn(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.downsample = nn.Identity()
        self.spike = spike_model(**neuron_params)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        identity = self.downsample(x)

        out += identity
        out = self.spike(out)
        return out
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super(ResidualBlock, self).__init__()
        
    def forward(self, x):
        raise NotImplementedError("This is a placeholder for the ANN Residual Block.")