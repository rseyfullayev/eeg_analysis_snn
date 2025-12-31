import torch
import torch.nn as nn
import snntorch as snn
from .decoders import ResNetDecoder, SpikingResNetDecoder
from ..layers.stem import BottleneckBlock, ClassifierHead
import snntorch.spikegen as spikegen
import torch.nn.functional as F

class SpikingUNet(nn.Module):
    def __init__(self, encoder, in_channels, num_classes, config, spike_model=snn.Leaky, **neuron_params):
        super(SpikingUNet, self).__init__()

        snn_params = neuron_params.copy()
        if spike_model.__name__  != "ALIF":
            snn_params['init_hidden'] = True
        self.encoding = config['data'].get('encoding_method', 'direct')
        self.num_timesteps = config['data'].get('num_timesteps', 10)
        self.encoder = encoder(in_channels, p_drop=config['model'].get('dropout', 0.2), spike_model=nn.SiLU)
        self.bottleneck = BottleneckBlock(512, p_drop=config['model'].get('dropout', 0.2), spike_model=spike_model, **snn_params)
        self.decoder = SpikingResNetDecoder(spike_model=spike_model, **snn_params)
        self.classifier = ClassifierHead(64, num_classes)

    def forward(self, x):
        if self.encoding == 'latency':
            x_static = x.mean(dim=0)
            x = spikegen.latency(x_static, num_steps=self.num_timesteps, tau=5, threshold=0.01, normalize=True, clip=True)

        elif self.encoding == 'rate': # Converges to Poisson encoding
            
            rand_map = torch.rand_like(x) 
            x = (x > rand_map).float()
        elif self.encoding == 'direct':
            pass
        else:
            raise ValueError(f"Unknown encoding method: {self.encoding}")
        
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skips)
        logits = self.classifier(x)
        
        return logits.mean(dim=0) 
    
class UNet(nn.Module):
    def __init__(self, encoder, in_channels, num_classes):
        super(UNet, self).__init__()

    def forward(self, x):
        raise NotImplementedError("This is a placeholder for the ANN UNet.")

class SpikingResNetClassifier(nn.Module):
    def __init__(self, encoder_backbone, num_classes=5, feature_dim=512, head_dim=128):
        super().__init__()

        self.encoder = encoder_backbone 
        self.num_classes = num_classes
        self.classifier = ClassifierHead(512, num_classes)
        self.supcon_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(feature_dim, head_dim, kernel_size=1)
        )

    def forward(self, x):
        features, _ = self.encoder(x)
        T, B, C, H, W = features.shape
        out = self.classifier(features)
        proj = self.supcon_head(features.view(T * B, C, H, W))
        embedding = F.normalize(proj.view(T * B, -1), dim=1)
        return out.mean(dim=0), embedding  # Mean over time dimension
