import torch.nn as nn
import snntorch as snn
from ..layers.residual_blocks import SpikingResBlock, ConvSpiking
from ..layers.stem import TemporalViTBlock, TemporalGCBlock

class SpikingResNet18Encoder(nn.Module):
    def __init__(self, in_channels, p_drop=0.1, vit_p_drop=0.25, vit=False, gc=False, spike_model=snn.Leaky, **neuron_params):
        super(SpikingResNet18Encoder, self).__init__()


        def _make_stage(in_c, out_c, stride, dilation, blocks, **params):
            layers = []
            layers.append(SpikingResBlock(in_c, 
                                          out_c, 
                                          stride=stride,
                                          p_drop=p_drop,
                                          spike_model=spike_model,
                                          **params))
            for _ in range(1, blocks):
                layers.append(SpikingResBlock(out_c, 
                                              out_c, 
                                              dilation=dilation,
                                              p_drop=p_drop,
                                              spike_model=spike_model,
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
        if spike_model.__name__ == 'ALIF':
            no_norm_layer_params['batch_norm'] = False

        last_layer_params = no_norm_layer_params.copy()
        if spike_model.__name__ == 'ALIF':
            last_layer_params['return_mem'] = True

        self.stage1 = _make_stage(20, 32, stride=2, dilation=1, blocks=2, **no_norm_layer_params)
        self.stage2 = _make_stage(32, 64, stride=2, dilation=1, blocks=3, **no_norm_layer_params)
        self.stage3 = _make_stage(64, 128, stride=2, dilation=2, blocks=3, **no_norm_layer_params)
        self.stage4 = _make_stage(128, 256, stride=2, dilation=4, blocks=2, **last_layer_params)

        '''

        self.layer1a = SpikingResBlock(20, 32, p_drop=p_drop, use_norm = True, spike_model=spike_model, **no_norm_layer_params)
        self.layer1b = SpikingResBlock(32, 32, p_drop=p_drop, use_norm = True, spike_model=spike_model, **no_norm_layer_params)
        
        self.layer2a = SpikingResBlock(32, 64, p_drop=p_drop, use_norm = True, stride=2, spike_model=spike_model, **no_norm_layer_params)
        self.layer2b = SpikingResBlock(64, 64, p_drop=p_drop, use_norm = True, spike_model=spike_model, **no_norm_layer_params)
        
        self.layer3a = SpikingResBlock(64, 128, p_drop=p_drop, use_norm = True, stride=2, spike_model=spike_model, **no_norm_layer_params)
        self.layer3b = SpikingResBlock(128, 128, p_drop=p_drop, use_norm = True, spike_model=spike_model, **no_norm_layer_params)

        
  
        self.layer4a = SpikingResBlock(128, 128, p_drop=p_drop, use_norm = True, stride=1, spike_model=spike_model, **last_layer_params)
        self.layer4b = SpikingResBlock(128, 128, p_drop=p_drop, use_norm = True, spike_model=spike_model, **last_layer_params)
        '''

        if vit:
            self.temporal = TemporalViTBlock(512, num_heads=8, p_drop=vit_p_drop)
        elif gc:
            self.temporal = TemporalGCBlock(512)
        else:
            self.temporal = nn.Identity()

    def forward(self, x):
        x = self.stem(x)
        '''
        s1 = self.layer1a(x)
        s1 = self.layer1b(s1)
        s2 = self.layer2a(s1)
        s2 = self.layer2b(s2)
        s3 = self.layer3a(s2)
        s3 = self.layer3b(s3)
        s4 = self.layer4a(s3)
        s4 = self.layer4b(s4)
        #s4 = self.temporal(s4)
        '''

        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s4 = self.temporal(s4)
        
        return s4, [s1, s2, s3]
    

