import torch.nn as nn
import snntorch as snn
import torch
from .residual_blocks import ConvBnSpiking

class SpikingUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, spike_model=snn.Leaky, **neuron_params):
        super(SpikingUpsampleBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.res = nn.Sequential(
            ConvBnSpiking(in_channels, out_channels, kernel_size=3, padding=1, spike_model=spike_model, **neuron_params),
            ConvBnSpiking(out_channels, out_channels, kernel_size=3, padding=1, spike_model=spike_model, **neuron_params)
        )
        

    def forward(self, x, x_skip):
        x = self.up(x)
        x = torch.cat([x, x_skip], dim=1)
        x = self.res(x)
        return x

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()

    def forward(self, x, x_skip):
        raise NotImplementedError("This is a placeholder for a non-spiking upsample block.")