import torch.nn as nn
import snntorch as snn
from ..layers.residual_blocks import *
from ..layers.upsampling_blocks import *

class SpikingResNetDecoder(nn.Module):
    def __init__(self, spike_model=snn.Leaky, **neuron_params):
        super(SpikingResNetDecoder, self).__init__()
        
        self.up1 = SpikingUpsampleBlock(
            in_channels=512, 
            skip_channels=256,
            id_channels=256, 
            out_channels=256, 
            spike_model=spike_model, **neuron_params
        )
        
        self.up2 = SpikingUpsampleBlock(
            in_channels=256, 
            skip_channels=128, 
            id_channels=128,
            out_channels=128, 
            spike_model=spike_model, **neuron_params
        )
        
        self.up3 = SpikingUpsampleBlock(
            in_channels=128, 
            skip_channels=64, 
            id_channels=64,
            out_channels=64, 
            spike_model=spike_model, **neuron_params
        )

        self.final_up = FinalUpBlock(
            in_channels=64, 
            out_channels=64, 
            spike_model=spike_model, **neuron_params
        )

    def forward(self, x, skips, skips_id):
        s1, s2, s3 = skips
        id1, id2, id3 = skips_id
        
        x = self.up1(x, s3, id3)
        x = self.up2(x, s2, id2)
        x = self.up3(x, s1, id1)
        
        x = self.final_up(x)
        return x

class ResNetDecoder(nn.Module):
    def __init__(self):
        super(ResNetDecoder, self).__init__()
        
    def forward(self, x, skips):
        raise NotImplementedError("This is a placeholder for the ANN ResNet Decoder.")