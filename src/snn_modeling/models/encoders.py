import torch.nn as nn
import snntorch as snn
from ..layers.residual_blocks import SpikingDualBlock
from ..layers.stem import StemLayer

class SpikingResNet18Encoder(nn.Module):
    def __init__(self, in_channels, p_drop=0.2, spike_model=snn.Leaky, **neuron_params):
        super(SpikingResNet18Encoder, self).__init__()
        self.stem = StemLayer(in_channels)
        
        self.layer1a = SpikingDualBlock(64, 64, p_drop=p_drop, spike_model=spike_model, **neuron_params)
        self.layer1b = SpikingDualBlock(64, 64, p_drop=p_drop, spike_model=spike_model, **neuron_params)
        
        self.layer2a = SpikingDualBlock(64, 128, p_drop=p_drop, stride=2, spike_model=spike_model, **neuron_params)
        self.layer2b = SpikingDualBlock(128, 128, p_drop=p_drop, spike_model=spike_model, **neuron_params)
        
        self.layer3a = SpikingDualBlock(128, 256, p_drop=p_drop, stride=2, spike_model=spike_model, **neuron_params)
        self.layer3b = SpikingDualBlock(256, 256, p_drop=p_drop, spike_model=spike_model, **neuron_params)
  
        self.layer4a = SpikingDualBlock(256, 512, p_drop=p_drop, stride=2, spike_model=spike_model, **neuron_params)
        self.layer4b = SpikingDualBlock(512, 512, p_drop=p_drop, spike_model=spike_model, **neuron_params)

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
        
        return s4, [s1, s2, s3]
    


class ResNet18Encoder(nn.Module):
    def __init__(self, in_channels):
        super(ResNet18Encoder, self).__init__()
    def forward(self, x):
        raise NotImplementedError("This is a placeholder for the ANN ResNet18 Encoder.")

# TODO: Implement ResNet50Encoder only if ResNet34Encoder is not sufficient for the task (BottleneckBlocks needed)