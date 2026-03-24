import torch.nn as nn
import snntorch as snn
from ..layers.residual_blocks import SpikingResBlock, ConvSpiking
from ..layers.stem import TemporalViTBlock, TemporalGCBlock
from ..layers.activations import instantiate_activation, resolve_activation, is_alif

class SpikingMobileNetEncoder(nn.Module):
    def __init__(self, in_channels, p_drop=0.1, vit_p_drop=0.25, vit=False, gc=False, spike_model=snn.Leaky, **neuron_params):
        super(SpikingMobileNetEncoder, self).__init__()
        
        spike_model = resolve_activation(spike_model)


        def _make_stage(in_c, out_c, stride, dilation, blocks, odconv, **params):
            layers = []
            layers.append(SpikingResBlock(in_c, 
                                          out_c, 
                                          stride=stride,
                                          p_drop=p_drop,
                                          spike_model=spike_model,
                                          odconv=odconv,
                                          **params))
            for _ in range(1, blocks):
                layers.append(SpikingResBlock(out_c, 
                                              out_c, 
                                              dilation=dilation,
                                              p_drop=p_drop,
                                              spike_model=spike_model,
                                              odconv=odconv,
                                              **params))
            return nn.Sequential(*layers)

        self.vit = vit
        self.stem = ConvSpiking(
            in_channels,
            20,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
            bias=False,
            spike_model=nn.SiLU,
            use_norm=True)
        
        no_norm_layer_params = neuron_params.copy()
        if is_alif(spike_model):
            no_norm_layer_params['batch_norm'] = False

        last_layer_params = no_norm_layer_params.copy()
        if is_alif(spike_model):
            last_layer_params['return_mem'] = True

        self.stage1 = _make_stage(20, 32, stride=2, dilation=1, blocks=2, odconv=False, **no_norm_layer_params)
        self.stage2 = _make_stage(32, 64, stride=2, dilation=1, blocks=3, odconv=False, **no_norm_layer_params)
        self.stage3 = _make_stage(64, 128, stride=2, dilation=2, blocks=3, odconv=True, **no_norm_layer_params)
        self.stage4 = _make_stage(128, 256, stride=2, dilation=4, blocks=2, odconv=True, **last_layer_params)


        if vit:
            self.temporal = TemporalViTBlock(512, num_heads=8, p_drop=vit_p_drop)
        elif gc:
            self.temporal = TemporalGCBlock(512)
        else:
            self.temporal = nn.Identity()

    def forward(self, x):
        x = self.stem(x)
      
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s4 = self.temporal(s4)
        
        return s4, [s1, s2, s3]
    

