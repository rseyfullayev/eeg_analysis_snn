import torch.nn as nn
import snntorch as snn
from .neurons import TimeDistributed, TemporalShift, TemporalOrderFix, ODConv2d
from torchvision.ops import StochasticDepth


class ConvSpiking(nn.Module):
    def __init__(self, in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 padding=0, 
                 groups=1,
                 dilation=1, 
                 bias=False, 
                 spike_model=snn.Leaky, 
                 use_norm = False,
                 odconv=False,
                 **neuron_params):
        super(ConvSpiking, self).__init__()
        #pad = dilation * (kernel_size - 1) // 2
        self.conv = TimeDistributed(ODConv2d(in_channels,
                                             out_channels,
                                             kernel_size=kernel_size, 
                                              stride=stride, 
                                              padding=padding,
                                              dilation=dilation,
                                              groups=groups) if odconv else 

                                              nn.Conv2d(in_channels, 
                                              out_channels, 
                                              kernel_size=kernel_size, 
                                              stride=stride, 
                                              padding=padding,
                                              dilation=dilation, 
                                              bias=bias, 
                                              groups=groups))
        
        layer_params = neuron_params.copy()
        if spike_model.__name__ == 'ALIF':
            layer_params['num_channels'] = out_channels

        self.norm = TemporalOrderFix(nn.InstanceNorm3d(out_channels, affine=True)) if use_norm else nn.Identity()

        self.spike = spike_model(**layer_params)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.spike(x)
        return x

class SpikingResBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 stride=1, 
                 dilation=1, 
                 p_drop=0.2, 
                 p_path=0.1, 
                 spike_model=snn.Leaky, 
                 use_norm = False,
                 odconv=False,
                 **neuron_params):
        
        super(SpikingResBlock, self).__init__()

        self.block1 = ConvSpiking(
            in_channels, 
            in_channels*2, 
            kernel_size=1,  
            bias=False, 
            spike_model=nn.Identity, 
            use_norm=use_norm,

        )

        self.block2 = ConvSpiking(
            in_channels*2, 
            in_channels*2, 
            kernel_size=3, 
            stride=stride, 
            padding=dilation,
            dilation=dilation,
            bias=False, 
            spike_model=spike_model, 
            groups=in_channels*2,
            use_norm=use_norm,
            odconv=odconv,
            **neuron_params
        )
        
        self.block3 = ConvSpiking(
            in_channels*2, 
            out_channels, 
            kernel_size=1,  
            bias=False, 
            spike_model=nn.Identity, 
            use_norm=use_norm,

        )
        self.tsm = TemporalShift(8)
        self.drop = TemporalOrderFix(nn.Dropout3d(p=p_drop))

        if stride != 1 or in_channels != out_channels:
            self.downsample = ConvSpiking(in_channels, 
                                          out_channels, 
                                          kernel_size=1, 
                                          stride=stride, 
                                          bias=False, 
                                          spike_model=nn.Identity, 
                                          use_norm=True)
        else:
            self.downsample = nn.Identity()

        layer_params = neuron_params.copy()
        if spike_model.__name__ == 'ALIF':
            layer_params['num_channels'] = out_channels
            
        self.final_spike = spike_model(**layer_params)
        self.drop_path = StochasticDepth(p=p_path, mode='row')

    def forward(self, x):
        
        identity = self.downsample(x)
        x = self.tsm(x)
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.drop(out)
        out = self.drop_path(out) + identity
        
        out = self.final_spike(out)
            
        return out
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super(ResidualBlock, self).__init__()
        
    def forward(self, x):
        raise NotImplementedError("This is a placeholder for the ANN Residual Block.")