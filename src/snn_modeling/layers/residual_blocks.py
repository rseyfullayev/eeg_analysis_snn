import torch.nn as nn
import snntorch as snn
import torch

class ConvBnSpiking(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, spike_model=snn.Leaky, **neuron_params):
        super(ConvBnSpiking, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.GroupNorm(1, out_channels) #nn.GroupNorm(1, out_channels) #nn.BatchNorm2d(out_channels)
        self.spike = spike_model(**neuron_params)

    def forward(self, x):
        if x.dim() == 5:
            T, B, C, H, W = x.shape
            x_flat = x.reshape(T * B, C, H, W)
            out = self.conv(x_flat)
            #out = self.bn(out)
            
            out_5d = out.reshape(T, B, out.shape[1], out.shape[2], out.shape[3])
            spikes = []
            for t in range(T):
                spikes.append(self.spike(out_5d[t]))
            return torch.stack(spikes, dim=0)
        else:
            x = self.conv(x)
            #x = self.bn(x)
            x = self.spike(x)
            return x
    
class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.GroupNorm(1, out_channels) #nn.GroupNorm(1, out_channels) #nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if x.dim() == 5:
            T, B, C, H, W = x.shape
        
            x_flat = x.reshape(T * B, C, H, W)
      
            out = self.conv(x_flat)
            #out = self.bn(out)
            
            _, C_out, H_out, W_out = out.shape
            out = out.reshape(T, B, C_out, H_out, W_out)
            return out
        else:
            return self.conv(x)

    
class SpikingResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, p_drop = 0.2, spike_model=snn.Leaky, **neuron_params):
        super(SpikingResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(1, out_channels) #nn.GroupNorm(1, out_channels) #nn.BatchNorm2d(out_channels)
        self.lif1 = spike_model(**neuron_params)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(1, out_channels) #nn.GroupNorm(1, out_channels) #nn.BatchNorm2d(out_channels)
        self.lif2 = spike_model(**neuron_params)
        self.drop = nn.Dropout2d(p=p_drop)
        if stride != 1 or in_channels != out_channels:
            self.downsample = self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(1, out_channels) #nn.GroupNorm(1, out_channels) #nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = nn.Identity()
        self.spike = spike_model(**neuron_params)

    def forward(self, x):
        T, B, C, H, W = x.shape
        
        x_flat = x.reshape(T * B, C, H, W)
        out = self.conv1(x_flat)
        
        out_5d = out.reshape(T, B, out.shape[1], out.shape[2], out.shape[3])
        
        spikes_1 = []
        for t in range(T):

            spikes_1.append(self.lif1(out_5d[t]))

        out_spikes = torch.stack(spikes_1, dim=0)
        

        out_flat = out_spikes.reshape(T * B, out_spikes.shape[2], out_spikes.shape[3], out_spikes.shape[4])
        
        out = self.conv2(out_flat)
        out = self.drop(out)
        
        identity = self.downsample(x_flat)

        out_5d = out.reshape(T, B, out.shape[1], out.shape[2], out.shape[3])
        
        spikes_2 = []
        for t in range(T):
            spikes_2.append(self.lif2(out_5d[t]))
            
        return torch.stack(spikes_2, dim=0)
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super(ResidualBlock, self).__init__()
        
    def forward(self, x):
        raise NotImplementedError("This is a placeholder for the ANN Residual Block.")