import torch
import torch.nn as nn
import snntorch as snn
from .decoders import ResNetDecoder, SpikingResNetDecoder
from ..layers.stem import BottleneckBlock, ClassifierHead, ProjectionHead
from ..layers.neurons import ALIF, TimeDistributed, SwiGLU
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
        encoder_mode = config['model'].get('encoder_mode', 'silu')
        encoder_spike_model = ALIF if encoder_mode == 'snn' else nn.SiLU
        self.encoder = encoder(
            in_channels,
            p_drop=config['model'].get('dropout', 0.2), 
            vit_p_drop=config['model'].get('vit_dropout', 0.25),
            vit=config['model'].get('vit_integration', False),
            gc=config['model'].get('gc_integration', False),
            spike_model=encoder_spike_model,
            **(snn_params if encoder_mode == 'snn' else {})
        )
        self.bottleneck = BottleneckBlock(128, p_drop=config['model'].get('dropout', 0.2), spike_model=spike_model, **snn_params)
        self.decoder = SpikingResNetDecoder(recurrent=config['model'].get('reccurent_decoder', False), spike_model=spike_model, **snn_params)
        self.classifier = ClassifierHead(64, num_classes)

    def forward(self, x, K=None):
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
    def __init__(self, encoder_backbone, num_classes=5, feature_dim=256, use_swiglu=False):
        super().__init__()

        self.encoder = encoder_backbone 
        self.num_classes = num_classes
        self.avg_pool = TimeDistributed(nn.AdaptiveAvgPool2d((1,1)))
        self.use_swiglu = use_swiglu
        
        # --- SwiGLU MIL Attention Heads (Optional) ---
        if self.use_swiglu:
            self.mil_attention = nn.Sequential(
                SwiGLU(feature_dim, p_drop=0.2),
                nn.Linear(feature_dim, 1, bias=False)
            )
        
        self.classifier = ProjectionHead(feature_dim, 128)
        
        

    def forward(self, x, K=None):
        features, _ = self.encoder(x)
        out = self.avg_pool(features).mean(dim=0).squeeze(-1).squeeze(-1)  # B x C x 1 x 1 -> B x C

        # === HYBRID MIL: Pre-Normalized RTFM Gating (or SwiGLU) ===
        if K is not None and K > 1:
            B_total, C_dim = out.shape
            B = B_total // K
            
            # Reshape to [Bags, Windows, Channels]
            out = out.view(B, K, C_dim)
            
            if self.use_swiglu:
                # 1. Compute Attention Scores using SwiGLU
                attn_scores = self.mil_attention(out)
                # 2. Normalize via Softmax across the K windows
                attn_weights = torch.softmax(attn_scores, dim=1)
                # 3. Aggregate windows via weighted sum
                out = torch.sum(out * attn_weights, dim=1)
            else:
                # --- Pre-Normalized RTFM Sieve ---
                # 1. Calculate unnormalized L2 magnitude of embeddings
                magnitudes = torch.linalg.norm(out, dim=-1) # [B, K]
                
                # 2. Extract Top-K (e.g., top 5) loudest bursts
                top_k_val = min(5, K) 
                _, topk_indices = torch.topk(magnitudes, k=top_k_val, dim=1) # [B, 5]
                
                master_vectors = []
                for b in range(B):
                    # 3. Gather highest magnitude vectors and average them BEFORE projection
                    loudest_embeds = out[b, topk_indices[b], :] # [5, C_dim]
                    master_vectors.append(loudest_embeds.mean(dim=0))
                    
                out = torch.stack(master_vectors, dim=0) # [B, C_dim]
        # ============================

        # The averaged Top-K (or SwiGLU-weighted) vector is passed to the Projection Head
        # which will apply F.normalize prior to SupCon!
        out = self.classifier(out)
        
        return out
