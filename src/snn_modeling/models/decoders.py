import torch.nn as nn
import snntorch as snn
from ..layers.residual_blocks import *
from ..layers.upsampling_blocks import *

class SpikingResNetDecoder(nn.Module):
    def __init__(self, spike_model=snn.Leaky, **neuron_params):
        super(SpikingResNetDecoder, self).__init__()
        
        self.up1 = SpikingUpsampleBlock(512, 256, spike_model=spike_model, **neuron_params)
        self.up2 = SpikingUpsampleBlock(256, 128, spike_model=spike_model, **neuron_params)
        self.up3 = SpikingUpsampleBlock(128, 64, spike_model=spike_model, **neuron_params)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            ConvBnSpiking(64, 64, kernel_size=3, padding=1, spike_model=spike_model, **neuron_params),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            ConvBnSpiking(64, 64, kernel_size=3, padding=1, spike_model=spike_model, **neuron_params)
        )

    def forward(self, x, skips):
        s1, s2, s3 = skips
        x = self.up1(x, s3)
        x = self.up2(x, s2)
        x = self.up3(x, s1)
        x = self.final_up(x)
        return x

class ResNetDecoder(nn.Module):
    def __init__(self):
        super(ResNetDecoder, self).__init__()
        
    def forward(self, x, skips):
        raise NotImplementedError("This is a placeholder for the ANN ResNet Decoder.")