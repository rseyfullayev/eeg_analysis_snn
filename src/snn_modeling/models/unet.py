import torch
import torch.nn as nn
import snntorch as snn
from .decoders import ResNetDecoder, SpikingResNetDecoder
from ..layers.stem import BottleneckBlock, ClassifierHead, ProjectionHead, TemporalGCBlock
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
    def __init__(self, encoder_backbone, num_classes=5, feature_dim=256, use_swiglu=False, use_batchnorm=False, use_dann=False, num_subjects=15):
        super().__init__()

        self.encoder = encoder_backbone 
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.use_swiglu = use_swiglu
        self.use_dann = use_dann
        self.num_subjects = num_subjects
        
        # --- SwiGLU MIL Attention Heads (Optional) ---
        if self.use_swiglu:
            self.mil_attention = nn.Sequential(
                SwiGLU(feature_dim, p_drop=0.2),
                nn.Linear(feature_dim, 1, bias=False)
            )
            self.cls_head = nn.Linear(128, num_classes)

        if self.use_dann:
            self.dann_head = nn.Sequential(
                GRL(alpha=1.0),
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, feature_dim // 2),
                nn.SiLU(),
                nn.Linear(feature_dim // 2, num_subjects)
            )
        
        # Auto-detect if encoder uses batchnorm to sync the ProjectionHead
        has_bn = any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) for m in self.encoder.modules())
        use_batchnorm = use_batchnorm or has_bn

        self.classifier = ProjectionHead(feature_dim, 128, use_batchnorm=use_batchnorm)
        
        
    def extract_features(self, x, K=None):
        """Extract backbone features (before projection head).
        
        Returns (B, feature_dim) tensor suitable for linear evaluation.
        """
        features, _ = self.encoder(x)
        # Spatiotemporal GAP: (T, B, C, H, W) -> (B, C)
        if features.dim() == 5:
            out = features.mean(dim=[0, 3, 4])  # mean over T, H, W
        else:
            out = features.mean(dim=[-2, -1])   # mean over H, W if already collapsed

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
                h_agg = torch.sum(out * attn_weights, dim=1)
                
                # Variational Info Bottleneck (VIB)
                z = h_agg # look into this (sampling may be requried)

                # Inverse attention baseline
                inv_weights = (1.0 - attn_weights)
                inv_weights = inv_weights / (inv_weights.sum(dim=1, keepdim=True) + 1e-8)
                h_base = torch.sum(out * inv_weights, dim=1)

                # Gram-Schmidt Orthogonalization
                proj = (torch.sum(z * h_base, dim=1, keepdim=True) / (torch.sum(h_base * h_base, dim=1, keepdim=True) + 1e-8)) * h_base
                out = z - proj
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

    def forward(self, x, K=None):
        if K is not None and K > 1:
            out = self.extract_features(x, K=K)
            logits = self.cls_head(out)
            return logits
            
        out = self.extract_features(x, K=K)
        out = self.classifier(out)
        return out
