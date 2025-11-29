import torch.nn as nn
import snntorch as snn
import torch
from .residual_blocks import ConvBnSpiking


class TimeDistributedUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(TimeDistributedUpsample, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.bn = nn.GroupNorm(1, out_channels) #nn.BatchNorm2d(out_channels)
    def forward(self, x):
        if x.dim() == 5:
            T, B, C, H, W = x.shape
            
            x_flat = x.reshape(T * B, C, H, W)
            

            out = self.up(x_flat)
            #out = self.bn(out)
            
            _, C_new, H_new, W_new = out.shape
            return out.reshape(T, B, C_new, H_new, W_new)
        else:

            return self.up(x)

class FinalUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, spike_model=snn.Leaky, **neuron_params):
        super(FinalUpBlock, self).__init__()
    
        self.up1 = TimeDistributedUpsample(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv1 = ConvBnSpiking(in_channels, in_channels, kernel_size=3, padding=1, spike_model=spike_model, **neuron_params)
    
        self.up2 = TimeDistributedUpsample(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv2 = ConvBnSpiking(out_channels, out_channels, kernel_size=3, padding=1, spike_model=spike_model, **neuron_params)

    def forward(self, x):
        
        x = self.up1(x)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = self.conv2(x)
        
        return x

class SpikingUpsampleBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, spike_model=snn.Leaky, **neuron_params):
        super(SpikingUpsampleBlock, self).__init__()
        self.upsample = TimeDistributedUpsample(in_channels, in_channels // 2)
        concat_channels = (in_channels // 2) + skip_channels
        self.conv1 = ConvBnSpiking(concat_channels, out_channels, kernel_size=3, padding=1, spike_model=spike_model, **neuron_params)
        self.conv2 = ConvBnSpiking(out_channels, out_channels, kernel_size=3, padding=1, spike_model=spike_model, **neuron_params)

    def forward(self, x, x_skip):
        x = self.upsample(x)
        x = torch.cat([x, x_skip], dim=2) 
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()

    def forward(self, x, x_skip):
        raise NotImplementedError("This is a placeholder for a non-spiking upsample block.")