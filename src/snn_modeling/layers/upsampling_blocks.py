import torch.nn as nn
import snntorch as snn
import torch
from .residual_blocks import ConvSpiking
from .neurons import TimeDistributed
from .activations import instantiate_activation, is_alif

class GatedSkip(nn.Module):
    def __init__(self, in_channels, spike_model=snn.Leaky, **neuron_params):
        super(GatedSkip, self).__init__()

        self.gate = TimeDistributed(nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        ))

        self.adapter = TimeDistributed(nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.SiLU(inplace=False)
        ))
        layer_params = neuron_params.copy()
        if is_alif(spike_model):
            layer_params['num_channels'] = in_channels
        self.spike = instantiate_activation(spike_model, **layer_params)
        nn.init.constant_(self.gate.module[0].bias, 2.0)

    def forward(self, x_skip):
        gate = self.gate(x_skip)
        adapted = self.adapter(x_skip)
        gated = adapted * gate
        gated = self.spike(gated)
        return gated


class UpsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, batch_norm=False, bias=False):
        super(UpsampleLayer, self).__init__()
        self.up = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size,
            stride=1,    
            padding=kernel_size//2, 
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-4) if batch_norm else nn.Identity()
        self.layer = TimeDistributed(nn.Sequential(
            self.up,
            self.conv,
            self.bn 
        ))

    def forward(self, x):
        return self.layer(x)

class SpikingUpsampleBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, upsample=True, spike_model=snn.Leaky, **neuron_params):
        super(SpikingUpsampleBlock, self).__init__()

        if upsample:
            self.upsample = UpsampleLayer(in_channels, in_channels // 2, kernel_size=3, stride=2)
            concat_channels = (in_channels // 2) + skip_channels
        else:
            self.upsample = nn.Identity()
            concat_channels = in_channels + skip_channels

        self.conv1 = ConvSpiking(
            concat_channels, 
            out_channels, 
            kernel_size=3, 
            padding=1, 
            spike_model=spike_model, 
            **neuron_params
        )
        self.gate = GatedSkip(skip_channels, spike_model=spike_model, **neuron_params)

        self.conv2 = ConvSpiking(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            padding=1, 
            spike_model=spike_model, 
            **neuron_params
        )

    def forward(self, x, x_skip):
        x = self.upsample(x)
        x_skip = self.gate(x_skip)

        x = torch.cat([x, x_skip], dim=2)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
class FinalUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FinalUpBlock, self).__init__()
    
        self.up1 = UpsampleLayer(in_channels, in_channels, kernel_size=3, stride=1, bias=True, batch_norm=True)
        self.act1 = nn.SiLU()

        self.up2 = UpsampleLayer(in_channels, out_channels, kernel_size=3, stride=1, bias=True)
        self.act2 = nn.SiLU()


    def forward(self, x):
        x = self.up1(x)
        x = self.act1(x)
        x = self.up2(x)
        x = self.act2(x)
        return x

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()

    def forward(self, x, x_skip):
        raise NotImplementedError("This is a placeholder for a non-spiking upsample block.")