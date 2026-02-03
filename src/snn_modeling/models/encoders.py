import torch.nn as nn
import snntorch as snn
from ..layers.residual_blocks import SpikingResBlock, ConvSpiking
from ..layers.stem import TemporalViTBlock, TemporalGCBlock

class SpikingResNet18Encoder(nn.Module):
    def __init__(self, in_channels, p_drop=0.1, vit_p_drop=0.25, vit=False, gc=False, spike_model=snn.Leaky, **neuron_params):
        super(SpikingResNet18Encoder, self).__init__()

        self.vit = vit
        self.stem = ConvSpiking(
            in_channels,
            32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            spike_model=nn.SiLU,
            use_norm=True)
        
        no_norm_layer_params = neuron_params.copy()
        if spike_model.__name__ == 'ALIF':
            no_norm_layer_params['batch_norm'] = False

        self.layer1a = SpikingResBlock(32, 32, p_drop=p_drop, use_norm = True, spike_model=spike_model, **no_norm_layer_params)
        #self.layer1b = SpikingResBlock(64, 64, p_drop=p_drop, use_norm = True, spike_model=spike_model, **no_norm_layer_params)
        
        self.layer2a = SpikingResBlock(32, 64, p_drop=p_drop, use_norm = True, stride=2, spike_model=spike_model, **no_norm_layer_params)
        self.layer2b = SpikingResBlock(64, 64, p_drop=p_drop, use_norm = True, spike_model=spike_model, **no_norm_layer_params)
        
        self.layer3a = SpikingResBlock(64, 128, p_drop=p_drop, use_norm = True, stride=2, spike_model=spike_model, **no_norm_layer_params)
        self.layer3b = SpikingResBlock(128, 128, p_drop=p_drop, use_norm = True, spike_model=spike_model, **no_norm_layer_params)

        last_layer_params = no_norm_layer_params.copy()
        if spike_model.__name__ == 'ALIF':
            last_layer_params['return_mem'] = True
  
        #self.layer4a = SpikingResBlock(256, 512, p_drop=p_drop, use_norm = True, stride=2, spike_model=spike_model, **last_layer_params)
        #self.layer4b = SpikingResBlock(512, 512, p_drop=p_drop, use_norm = True, spike_model=spike_model, **last_layer_params)

        if vit:
            self.temporal = TemporalViTBlock(512, num_heads=8, p_drop=vit_p_drop)
        elif gc:
            self.temporal = TemporalGCBlock(512)
        else:
            self.temporal = nn.Identity()

    def forward(self, x):
        x = self.stem(x)
        s1 = self.layer1a(x)
        #s1 = self.layer1b(s1)
        s2 = self.layer2a(s1)
        s2 = self.layer2b(s2)
        s3 = self.layer3a(s2)
        s3 = self.layer3b(s3)
        #s4 = self.layer4a(s3)
        #s4 = self.layer4b(s4)
        #s4 = self.temporal(s4)
        return s3, [s1, s2] #s4, [s1, s2, s3]
    


class ResNet18Encoder(nn.Module):
    def __init__(self, in_channels):
        super(ResNet18Encoder, self).__init__()
    def forward(self, x):
        raise NotImplementedError("This is a placeholder for the ANN ResNet18 Encoder.")

# TODO: Implement ResNet50Encoder only if ResNet34Encoder is not sufficient for the task (BottleneckBlocks needed)