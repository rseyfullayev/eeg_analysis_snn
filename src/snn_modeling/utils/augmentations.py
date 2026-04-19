import torch
import torch.nn as nn
from torchvision.transforms import RandomErasing

class DyTNorm(nn.Module):
    """
    Instance-Adaptive Dynamic Tanh Normalization.
    Input: [B, T, C, H, W] or [T, C, H, W]

    Adapted from:

    @inproceedings{Zhu2025DyT,
    title={Transformers without Normalization},
    author={Zhu, Jiachen and Chen, Xinlei and He, Kaiming and LeCun, Yann and Liu, Zhuang},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2025}
    }


    """
    def __init__(self, gain=3.0, percentile=0.98):
        super().__init__()
        self.gain = gain
        self.percentile = percentile

    def forward(self, x):
        if x.dim() == 5:
            B = x.shape[0]
            flat = x.view(B, -1).abs()
            p_val = torch.quantile(flat, self.percentile, dim=1, keepdim=True)
            p_val = p_val.view(B, 1, 1, 1, 1)
        else:
            flat = x.abs().flatten()
            p_val = torch.quantile(flat, self.percentile)
        
        p_safe = torch.maximum(p_val, torch.tensor(1e-6, device=x.device))
        return torch.tanh(x / p_safe * self.gain)
    
class GaussianNoise(nn.Module):
    def __init__(self, std=0.05):
        super().__init__()
        self.std = std

    def forward(self, x):
        if not self.training: return x
        noise = torch.randn_like(x) * self.std
        return x + noise

class SignalJitter(nn.Module):
    def __init__(self, lower=0.8, upper=1.2):
        super().__init__()
        self.lower = lower
        self.upper = upper

    def forward(self, x):
        if not self.training: return x
        jitter = torch.empty(1).uniform_(self.lower, self.upper).to(x.device)
        return x * jitter
    
class FrequencyDropout(nn.Module):
    """
    Randomly drops entire frequency bands (channels).
    Input: [B, T, C, H, W] or [T, C, H, W]
    """
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training: return x
        if x.dim() == 5:
            # x: [B, T, C, H, W]
            C = x.shape[2]
            mask = (torch.rand(C, device=x.device) > self.p).float()
            mask = mask.view(1, 1, C, 1, 1)
        else:
            # x: [T, C, H, W]
            C = x.shape[1]
            mask = (torch.rand(C, device=x.device) > self.p).float()
            mask = mask.view(1, C, 1, 1)
        return x * mask

class VideoTemporalMasking(nn.Module):
    """
    Masks out a CONTIGUOUS block of frames.
    Simulates a sustained sensor disconnect or a wiped-out artifact.
    Input shape: [B, T, C, H, W] or [T, C, H, W]
    """
    def __init__(self, p=0.3, max_mask_len=5):
        super().__init__()
        self.p = p
        self.max_mask_len = max_mask_len

    def forward(self, x):
        if not self.training: return x
        
        if torch.rand(1).item() < self.p:
            x = x.clone()  # Safe for SupCon multi-view
            
            if x.dim() == 5:
                # x: [B, T, C, H, W]
                T = x.shape[1]
                mask_len = torch.randint(1, min(self.max_mask_len + 1, T), (1,)).item()
                start_idx = torch.randint(0, T - mask_len + 1, (1,)).item()
                x[:, start_idx : start_idx + mask_len, :, :, :] = 0.0
            else:
                # x: [T, C, H, W]
                T = x.shape[0]
                mask_len = torch.randint(1, min(self.max_mask_len + 1, T), (1,)).item()
                start_idx = torch.randint(0, T - mask_len + 1, (1,)).item()
                x[start_idx : start_idx + mask_len, :, :, :] = 0.0
            
        return x


class VideoRandomErasing(nn.Module):
    """
    Applies Cutout/Erasing to frames. 
    Input: [B, T, C, H, W] or [T, C, H, W]
    """
    def __init__(self, p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)):
        super().__init__()
        self.p = p
        self.eraser = RandomErasing(p=1.0, scale=scale, ratio=ratio, value=0, inplace=False)

    def forward(self, x):
        if not self.training: return x
        if torch.rand(1).item() < self.p:
            if x.dim() == 5: # x: [B, T, C, H, W]
                frame_0 = x[:, 0:1].clone()   
            else: # x: [T, C, H, W]
                frame_0 = x[0:1].clone() 

            erased_frame = self.eraser(frame_0) 
            
            # Detect erased region by comparing before/after (not just == 0)
            mask = (frame_0 != erased_frame)
            
            x = x.clone()
            x = x.masked_fill_(mask, 0.0)

        return x

class SpatialDropout(nn.Module):
    """
    Randomly zeroes out specific spatial locations (electrodes) on the topographic map.
    This simulates dropped connections or varying impedances at specific electrode locations.
    The dropout mask is consistent across all time steps and frequency bands for a given clip.
    Input shape: [B, T, C, H, W] or [T, C, H, W]
    """
    def __init__(self, p=0.05):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training: return x
        
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            # Drop each electrode (H, W) independently per sample in the batch
            mask = (torch.rand(B, 1, 1, H, W, device=x.device) > self.p).float()
        else:
            # x: [T, C, H, W]
            T, C, H, W = x.shape
            mask = (torch.rand(1, 1, H, W, device=x.device) > self.p).float()
            
        return x * mask

class TemporalMix(nn.Module):
    """
    Temporal Mixup: Mixes multiple videos along the temporal dimension.
    Input: [B, T, C, H, W]
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        if not self.training: return x, y
        
        B, T, C, H, W = x.shape
        x_mixed = x.clone()
        
        classes = y.unique()
        for cls in classes:
            idxs = (y == cls).nonzero(as_tuple=True)[0]
            n_samples = len(idxs)
            if n_samples < 2: 
                continue
            for t in range(T):
                shuffled_indices = idxs[torch.randperm(n_samples)]
                x_mixed[idxs, t] = x[shuffled_indices, t]

        return x_mixed, y
