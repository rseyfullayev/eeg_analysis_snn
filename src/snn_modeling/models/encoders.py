import torch.nn as nn
import snntorch as snn
from ..layers.residual_blocks import SpikingResidualBlock
from ..layers.stem import StemLayer

class SpikingResNet18Encoder(nn.Module):
    def __init__(self, in_channels, p_drop=0.2, spike_model=snn.Leaky, **neuron_params):
        super(SpikingResNet18Encoder, self).__init__()
        self.stem = StemLayer(in_channels)
        
        self.layer1 = nn.Sequential(
            SpikingResidualBlock(64, 64, p_drop=p_drop, spike_model=spike_model, **neuron_params),
            SpikingResidualBlock(64, 64, p_drop=p_drop, spike_model=spike_model, **neuron_params)
        )
        self.layer2 = nn.Sequential(
            SpikingResidualBlock(64, 128, p_drop=p_drop, stride=2, spike_model=spike_model, **neuron_params),
            SpikingResidualBlock(128, 128, p_drop=p_drop, spike_model=spike_model, **neuron_params)
        )
        self.layer3 = nn.Sequential(
            SpikingResidualBlock(128, 256, p_drop=p_drop, stride=2, spike_model=spike_model, **neuron_params),
            SpikingResidualBlock(256, 256, p_drop=p_drop, spike_model=spike_model, **neuron_params)
        )
        self.layer4 = nn.Sequential(
            SpikingResidualBlock(256, 512, p_drop=p_drop, stride=2, spike_model=spike_model, **neuron_params),
            SpikingResidualBlock(512, 512, p_drop=p_drop, spike_model=spike_model, **neuron_params)
        )
    def forward_layers(self, x):

        s1 = self.layer1(x)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        return s4, [s1,s2,s3]
    
    def forward(self, x):
        x = self.stem(x)
        return self.forward_layers(x)
    
class SpikingResNet34Encoder(nn.Module):
    def __init__(self, in_channels, p_drop=0.2, spike_model=snn.Leaky, **neuron_params):
        super(SpikingResNet34Encoder, self).__init__()
        self.stem = StemLayer(in_channels)
        
        self.layer1 = nn.Sequential(
            SpikingResidualBlock(64, 64, p_drop=p_drop, spike_model=spike_model, **neuron_params),
            SpikingResidualBlock(64, 64, p_drop=p_drop, spike_model=spike_model, **neuron_params),
            SpikingResidualBlock(64, 64, p_drop=p_drop, spike_model=spike_model, **neuron_params)
        )
        self.layer2 = nn.Sequential(
            SpikingResidualBlock(64, 128, p_drop=p_drop, stride=2, spike_model=spike_model, **neuron_params),
            SpikingResidualBlock(128, 128, p_drop=p_drop, spike_model=spike_model, **neuron_params),
            SpikingResidualBlock(128, 128, p_drop=p_drop, spike_model=spike_model, **neuron_params),
            SpikingResidualBlock(128, 128, p_drop=p_drop, spike_model=spike_model, **neuron_params)
        )
        self.layer3 = nn.Sequential(
            SpikingResidualBlock(128, 256, p_drop=p_drop, stride=2, spike_model=spike_model, **neuron_params),
            SpikingResidualBlock(256, 256, p_drop=p_drop, spike_model=spike_model, **neuron_params),
            SpikingResidualBlock(256, 256, p_drop=p_drop, spike_model=spike_model, **neuron_params),
            SpikingResidualBlock(256, 256, p_drop=p_drop, spike_model=spike_model, **neuron_params),
            SpikingResidualBlock(256, 256, p_drop=p_drop, spike_model=spike_model, **neuron_params),
            SpikingResidualBlock(256, 256, p_drop=p_drop, spike_model=spike_model, **neuron_params)
        )
        self.layer4 = nn.Sequential(
            SpikingResidualBlock(256, 512, p_drop=p_drop, stride=2, spike_model=spike_model, **neuron_params),
            SpikingResidualBlock(512, 512, p_drop=p_drop, spike_model=spike_model, **neuron_params),
            SpikingResidualBlock(512, 512, p_drop=p_drop, spike_model=spike_model, **neuron_params)
        )
    def forward_layers(self, x):
        s1 = self.layer1(x)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        return s4, [s1,s2,s3]
    
    def forward(self, x):
        x = self.stem(x)
        return self.forward_layers(x)

class ResNet18Encoder(nn.Module):
    def __init__(self, in_channels):
        super(ResNet18Encoder, self).__init__()
    def forward(self, x):
        raise NotImplementedError("This is a placeholder for the ANN ResNet18 Encoder.")

class ResNet34Encoder(nn.Module):
    def __init__(self, in_channels):
        super(ResNet34Encoder, self).__init__()
    def forward(self, x):
        raise NotImplementedError("This is a placeholder for the ANN ResNet18 Encoder.")
    
# TODO: Implement ResNet50Encoder only if ResNet34Encoder is not sufficient for the task (BottleneckBlocks needed)