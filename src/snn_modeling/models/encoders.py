import torch.nn as nn
import snntorch as snn
from ..layers.residual_blocks import SpikingResBlock
from ..layers.stem import StemLayer, TemporalViTBlock

class SpikingResNet18Encoder(nn.Module):
    def __init__(self, in_channels, p_drop=0.1, vit_p_drop=0.25, vit=True, spike_model=snn.Leaky, **neuron_params):
        super(SpikingResNet18Encoder, self).__init__()

        self.vit = vit
        self.stem = StemLayer(in_channels)
        
        self.layer1a = SpikingResBlock(64, 64, p_drop=p_drop, use_norm = True, spike_model=spike_model, **neuron_params)
        self.layer1b = SpikingResBlock(64, 64, p_drop=p_drop, use_norm = True, spike_model=spike_model, **neuron_params)
        
        self.layer2a = SpikingResBlock(64, 128, p_drop=p_drop, use_norm = True, stride=2, spike_model=spike_model, **neuron_params)
        self.layer2b = SpikingResBlock(128, 128, p_drop=p_drop, use_norm = True, spike_model=spike_model, **neuron_params)
        
        self.layer3a = SpikingResBlock(128, 256, p_drop=p_drop, use_norm = True, stride=2, spike_model=spike_model, **neuron_params)
        self.layer3b = SpikingResBlock(256, 256, p_drop=p_drop, use_norm = True, spike_model=spike_model, **neuron_params)
  
        self.layer4a = SpikingResBlock(256, 512, p_drop=p_drop, use_norm = True, stride=2, spike_model=spike_model, **neuron_params)
        self.layer4b = SpikingResBlock(512, 512, p_drop=p_drop, use_norm = True, spike_model=spike_model, **neuron_params)

        if vit:
            self.temporal_vit = TemporalViTBlock(512, num_heads=8, p_drop=vit_p_drop)
        else:
            self.temporal_vit = nn.Identity()

    def forward(self, x):
        x = self.stem(x)
        s1 = self.layer1a(x)
        s1 = self.layer1b(s1)
        s2 = self.layer2a(s1)
        s2 = self.layer2b(s2)
        s3 = self.layer3a(s2)
        s3 = self.layer3b(s3)
        s4 = self.layer4a(s3)
        s4 = self.layer4b(s4)
        s4 = self.temporal_vit(s4)
        return s4, [s1, s2, s3]
    


class ResNet18Encoder(nn.Module):
    def __init__(self, in_channels):
        super(ResNet18Encoder, self).__init__()
    def forward(self, x):
        raise NotImplementedError("This is a placeholder for the ANN ResNet18 Encoder.")

# TODO: Implement ResNet50Encoder only if ResNet34Encoder is not sufficient for the task (BottleneckBlocks needed)