import torch.nn as nn
import snntorch as snn
from .decoders import ResNetDecoder
from ..layers.stem import BottleneckBlock, ClassifierHead

class SpikingUNet(nn.Module):
    def __init__(self, encoder, num_classes, spike_model=snn.Leaky, **neuron_params):
        super(SpikingUNet, self).__init__()
        self.encoder = encoder
        self.bottleneck = BottleneckBlock(512, spike_model=spike_model, **neuron_params)
        self.decoder = ResNetDecoder(spike_model=spike_model, **neuron_params)
        self.classifier = ClassifierHead(64, num_classes)

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skips)
        x = self.classifier(x)
        return x

