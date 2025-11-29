import torch.nn as nn
import snntorch as snn
import torch

class ConvBnSpiking(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, spike_model=snn.Leaky, **neuron_params):
        super(ConvBnSpiking, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.GroupNorm(1, out_channels)
        self.spike = spike_model(**neuron_params)

    def forward(self, x):
        if x.dim() == 5:
            T, B, C, H, W = x.shape
            x_flat = x.reshape(T * B, C, H, W)
            out = self.conv(x_flat)
            out = self.bn(out)
            
            out_5d = out.reshape(T, B, out.shape[1], out.shape[2], out.shape[3])
            spikes = []
            for t in range(T):
                spikes.append(self.spike(out_5d[t]))
            return torch.stack(spikes, dim=0)
        else:
            x = self.conv(x)
            x = self.bn(x)
            x = self.spike(x)
            return x
    
class ConvSpiking(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, spike_model=snn.Leaky, **neuron_params):
        super(ConvSpiking, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.spike = spike_model(**neuron_params)

    def forward(self, x):
        if x.dim() == 5:
            T, B, C, H, W = x.shape
            x_flat = x.reshape(T * B, C, H, W)
            out = self.conv(x_flat)
            
            out_5d = out.reshape(T, B, out.shape[1], out.shape[2], out.shape[3])
            spikes = []
            for t in range(T):
                spikes.append(self.spike(out_5d[t]))
            return torch.stack(spikes, dim=0)
        else:
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
        
        self.drop = nn.Dropout2d(p=p_drop)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True)
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        if x.dim() == 5:
            T, B, C, H, W = x.shape
            x_flat = x.reshape(T * B, C, H, W)
            identity = self.downsample(x_flat)
            _, C_out, H_out, W_out = identity.shape
            identity = identity.reshape(T, B, C_out, H_out, W_out)
        else:
            identity = self.downsample(x)

        out = self.block1(x)
        
        if out.dim() == 5:
            T, B, C, H, W = out.shape
            out = out.reshape(T * B, C, H, W)
            out = self.drop(out)
            out = out.reshape(T, B, C, H, W)
        else:
            out = self.drop(out)

        out = self.block2(out)
            
        return out, identity
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super(ResidualBlock, self).__init__()
        
    def forward(self, x):
        raise NotImplementedError("This is a placeholder for the ANN Residual Block.")