import torch
import torch.nn as nn
import snntorch as snn
from .decoders import ResNetDecoder, SpikingResNetDecoder
from ..layers.stem import BottleneckBlock, ClassifierHead, ProjectionHead, TemporalGCBlock, WindowReRanker, VIBLayer
from ..layers.neurons import ALIF, TimeDistributed, SwiGLU
import snntorch.spikegen as spikegen
import torch.nn.functional as F

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GRL(nn.Module):
    def __init__(self, alpha=1.0):
        super(GRL, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

class SpikingUNet(nn.Module):
    def __init__(self, encoder_backbone, decoder_backbone, in_channels, num_classes, encoding_method='direct', num_timesteps=32):
        super(SpikingUNet, self).__init__()

        self.encoding = encoding_method
        self.num_timesteps = num_timesteps
        
        self.encoder = encoder_backbone
        
        # We assume the decoder is also pre-built from the new Hydra configs!
        self.bottleneck = BottleneckBlock(128, p_drop=0.2, spike_model=nn.SiLU) # Standardizing bottleneck to standard for simplicity right now unless we want to inject it too!
        self.decoder = decoder_backbone

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

class SpikingMobileNetProjector(nn.Module):
    def __init__(self, encoder_backbone, 
                 num_classes=5, feature_dim=256, 
                 use_swiglu=False, use_batchnorm=False, 
                 use_dann=False, use_vib=True, use_subj=True, num_subjects=15,
                 max_windows=1000):
        super().__init__()
        self.encoder = encoder_backbone
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.use_swiglu = use_swiglu
        self.use_dann = use_dann
        self.use_vib = use_vib
        self.use_subj = use_subj
        self.num_subjects = num_subjects
        
        # --- SwiGLU MIL Attention Heads (Optional) ---
        if self.use_swiglu:
            if self.use_vib:
                self.vib = VIBLayer(feature_dim)
                cls_in_dim = feature_dim // 4
            else:
                cls_in_dim = feature_dim
            
            self.reranker = WindowReRanker(feature_dim, max_windows=max_windows)
        
            self.cls_head = nn.Sequential(
                nn.Linear(cls_in_dim, cls_in_dim),
                nn.SiLU(),
                nn.Linear(cls_in_dim, num_classes)
            )

        if self.use_dann:
            # GRL Adversarial Head on z_emo: Forces emotion latent to contain NO subject information
            self.dann_head = nn.Sequential(
                GRL(alpha=1.0),
                nn.Linear(cls_in_dim if self.use_swiglu else feature_dim, cls_in_dim if self.use_swiglu else feature_dim),
                nn.SiLU(),
                nn.Linear(cls_in_dim if self.use_swiglu else feature_dim, num_subjects)
            )
            
            if self.use_subj:
                # Explicit Subject Classification Head on h_dmn: Explicitly pulls subject variance into h_dmn
                self.subj_head = nn.Sequential(
                    nn.LayerNorm(feature_dim),
                    nn.Linear(feature_dim, feature_dim // 2),
                    nn.SiLU(),
                    nn.Linear(feature_dim // 2, num_subjects)
                )
        
        # Auto-detect if encoder uses batchnorm to sync the ProjectionHead
        has_bn = any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) for m in self.encoder.modules())
        use_batchnorm = use_batchnorm or has_bn

        self.classifier = ProjectionHead(feature_dim, 128, use_batchnorm=use_batchnorm)
        
        
        
    def extract_features(self, x, K=None, mask=None):
        """Extract backbone features (before projection head).
        
        Returns (B, feature_dim) tensor suitable for linear evaluation.
        """
        if x.dim() == 3:
            # Precomputed features: (B, W, C_dim)
            out = x
            B, K, C_dim = out.shape
        else:
            features, _ = self.encoder(x)
            # Spatiotemporal GAP: (T, B, C, H, W) -> (B, C)
            if features.dim() == 5:
                out = features.mean(dim=[0, 3, 4])  # mean over T, H, W
            else:
                out = features.mean(dim=[-2, -1])   # mean over H, W if already collapsed

        # === HYBRID MIL: Pre-Normalized RTFM Gating (or SwiGLU) ===
        if K is not None and K > 1:
            if x.dim() != 3:
                B_total, C_dim = out.shape
                B = B_total // K
                # Reshape to [Bags, Windows, Channels]
                out = out.view(B, K, C_dim)
            
            if self.use_swiglu:
                attn_weights = self.reranker(out, mask=mask)
                h_emo = torch.sum(out * attn_weights, dim=1)
   
                inv_weights = (1.0 - attn_weights)
                if mask is not None:
                    inv_weights = inv_weights.masked_fill(~mask.unsqueeze(-1), 0.0)
                inv_weights = inv_weights / (inv_weights.sum(dim=1, keepdim=True) + 1e-8)
                h_dmn = torch.sum(out * inv_weights, dim=1)

                # Gram-Schmidt Orthogonalization
                proj = (torch.sum(h_emo * h_dmn, dim=1, keepdim=True) / (torch.sum(h_dmn * h_dmn, dim=1, keepdim=True) + 1e-8)) * h_dmn
                if self.use_subj:
                    h_emo = h_emo - proj
                
                # Attention Entropy: -sum(p * log(p))
                attn_entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-8), dim=1).mean()
                return h_dmn, h_emo, attn_entropy
            else:
                # --- Pre-Normalized RTFM Sieve ---
                # 1. Calculate unnormalized L2 magnitude of embeddings
                magnitudes = torch.linalg.norm(out, dim=-1) # [B, K]
                
                # 2. Extract Top-K (e.g., top 5) loudest bursts
                top_k_val = min(5, K) 
                _, topk_indices = torch.topk(magnitudes, k=top_k_val, dim=1) # [B, 5]
                
                gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, out.size(-1))
                master_vectors = torch.gather(out, dim=1, index=gather_indices) # [B, min(5, K), C_dim]
                    
                out = torch.stack(master_vectors, dim=0).mean(dim=1) # [B, C_dim]
        # ============================

        return out

    def forward(self, x, K=None, mask=None):
        if K is not None and K > 1:
            h_dmn, h_emo, attn_entropy = self.extract_features(x, K=K, mask=mask)
            
            if self.use_swiglu and getattr(self, 'use_vib', True):
                z_emo, mu, logvar = self.vib(h_emo)
            else:
                z_emo, mu, logvar = h_emo, None, None
                
            logits = self.cls_head(z_emo)
            
            if self.use_dann:
                dann_logits = self.dann_head(z_emo)  # GRL applied to z_emo
                
                if getattr(self, 'use_subj', True):
                    subj_logits = self.subj_head(h_dmn)  # No GRL, explicit routing for h_dmn
                else:
                    subj_logits = None
                    
                return logits, dann_logits, subj_logits, mu, logvar, h_emo, h_dmn, attn_entropy
            return logits, mu, logvar, h_emo, h_dmn, attn_entropy
            
        out = self.extract_features(x, K=K, mask=mask)
        out = self.classifier(out)
        return out
