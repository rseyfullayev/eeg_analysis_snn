import torch.nn as nn
import snntorch as snn
import torch
from .neurons import ALIF, TimeDistributed
    
class ConvSpiking(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, spike_model=snn.Leaky, **neuron_params):
        super(ConvSpiking, self).__init__()
        self.conv = TimeDistributed(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        layer_params = neuron_params.copy()
        if spike_model.__name__ == 'ALIF':
            layer_params['num_channels'] = out_channels
        self.spike = spike_model(**layer_params)

    def forward(self, x):
        x = self.conv(x)
        x = self.spike(x)
        return x

class SpikingDualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, p_drop=0.2, spike_model=snn.Leaky, **neuron_params):
        super(SpikingDualBlock, self).__init__()

        self.block1 = ConvSpiking(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1, 
            bias=False, 
            spike_model=spike_model, 
            **neuron_params
        )
        
        self.block2 = ConvSpiking(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            padding=1, 
            bias=False, 
            spike_model=spike_model, 
            **neuron_params
        )
        
        self.drop = TimeDistributed(nn.Dropout2d(p=p_drop))

        if stride != 1 or in_channels != out_channels:
            self.downsample = TimeDistributed(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True))
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.block1(x)
        out = self.drop(out)
        out += identity
        out = self.block2(out)
            
        return out
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super(ResidualBlock, self).__init__()
        
    def forward(self, x):
        raise NotImplementedError("This is a placeholder for the ANN Residual Block.")