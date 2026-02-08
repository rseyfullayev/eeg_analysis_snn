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
        jitter = torch.empty(1).uniform_(0.8, 1.2).to(x.device)
        return x * jitter
    
class FrequencyDropout(nn.Module):
    """
    Randomly drops entire frequency bands (channels).
    """
    def __init__(self, p=0.2):
        super().__init__()
        self.drop = nn.Dropout3d(p=p)

    def forward(self, x):
        if not self.training: return x
        x = self.drop(x)
        return x
        
class TemporalMasking(nn.Module):
    """
    Randomly masks out entire frames in the temporal dimension.
    """
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training: return x
        _,T,_,_ = x.shape
        if torch.rand(1) < self.time_mask_prob:
            n_msk = torch.randint(1,4,(1,)).item()
            ind = torch.randperm(T)[:n_msk]   
            x[:,ind,:,:] = 0.0

class VideoRandomErasing(nn.Module):
    """
    Applies Cutout/Erasing to frames. 
    Can erase the same spot in all frames (Spatial Consistency) 
    or different spots (Temporal Chaos).
    """
    def __init__(self, p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), consistent=False):
        super().__init__()
        self.consistent = consistent
        self.eraser = RandomErasing(p=p, scale=scale, ratio=ratio, value=0, inplace=False)

    def forward(self, x):
        if not self.training: return x
        
        B, T, C, H, W = x.shape

        x_flat = x.view(B * T, C, H, W)
        out = self.eraser(x_flat)

        return out.view(B, T, C, H, W)

class TemporalMix(nn.Module):
    """
    Temporal Mixup: Mixes multiple videos along the temporal dimension.
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

