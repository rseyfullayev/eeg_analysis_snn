import torch
import torch.nn as nn
import torch.fft
import numpy as np
from tqdm import tqdm
from collections import defaultdict
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

# ======================== Fourier DA Mixup (3D) ======================== #

def compute_subject_variance_map(dataset, max_samples_per_subject=200, device='cuda'):
    from collections import defaultdict
    subject_magnitudes = defaultdict(list)
    print("[FourierMixup] Phase 1: Computing subject magnitude centroids in 3D...")
    
    for idx in tqdm(range(len(dataset)), desc="Scanning FFTs"):
        sample = dataset[idx]
        if len(sample) == 8:
            video = sample[0]
            subject_idx = sample[4]
        else:
            video = sample[0]
            subject_idx = dataset.samples[idx][3]
            
        if subject_idx == -1: continue
        if len(subject_magnitudes[subject_idx]) >= max_samples_per_subject: continue
            
        if video.dim() == 5:
            volume = video[0].mean(dim=1)  # (T, H, W)
        elif video.dim() == 4:
            volume = video.mean(dim=1)  # (T, H, W)
        else:
            continue
            
        volume = volume.to(device).float()
        
        fft_3d = torch.fft.fftn(volume)
        fft_shifted = torch.fft.fftshift(fft_3d)
        magnitude = torch.abs(fft_shifted)
        
        subject_magnitudes[subject_idx].append(magnitude.cpu())
    
    subject_ids_sorted = sorted(subject_magnitudes.keys())
    centroids = []
    donor_magnitudes = []
    for subj_id in subject_ids_sorted:
        mags = torch.stack(subject_magnitudes[subj_id])
        centroid = mags.mean(dim=0)  # (T, H, W)
        centroids.append(centroid)
        donor_magnitudes.append((subj_id, centroid))
    
    if len(centroids) < 2:
        print("[FourierMixup] Warning: fewer than 2 subjects found, returning uniform variance map.")
        shape = centroids[0].shape if centroids else (16, 32, 32)
        return torch.ones(shape), shape, donor_magnitudes
        
    centroids = torch.stack(centroids)
    variance_map = centroids.var(dim=0)
    variance_map = variance_map / (variance_map.max() + 1e-8)
    shape = variance_map.shape
    print(f"[FourierMixup] Phase 1 complete: {len(centroids)} subjects, shape={shape}")
    
    return variance_map, shape, donor_magnitudes

def compute_optimal_sigma(variance_map, retention_ratio=0.95, max_iter=50, tol=1e-6):
    T, H, W = variance_map.shape
    total_energy = variance_map.sum().item()
    target_energy = retention_ratio * total_energy
    
    ct, cy, cx = T // 2, H // 2, W // 2
    tt, yy, xx = torch.meshgrid(torch.arange(T, dtype=torch.float32),
                                torch.arange(H, dtype=torch.float32),
                                torch.arange(W, dtype=torch.float32), indexing='ij')
    dist_sq = (tt - ct) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
    
    sigma_min = 0.5
    sigma_max = float(max(T, H, W))
    
    print(f"[FourierMixup] Phase 2: Binary search for σ (target retention={retention_ratio:.2f})...")
    for i in range(max_iter):
        sigma = (sigma_min + sigma_max) / 2.0
        gaussian = torch.exp(-dist_sq / (2.0 * sigma ** 2))
        filtered_energy = (variance_map * gaussian).sum().item()
        
        if abs(filtered_energy - target_energy) / (total_energy + 1e-8) < tol: break
        if filtered_energy < target_energy: sigma_min = sigma
        else: sigma_max = sigma
    
    print(f"[FourierMixup] Phase 2 complete: optimal σ = {sigma:.2f}")
    return sigma

class FourierMixup(nn.Module):
    def __init__(self, sigma, shape, p=0.5, beta_alpha=1.0, donor_magnitudes=None):
        super().__init__()
        self.p = p
        self.beta_alpha = beta_alpha
        self.sigma = sigma
        self.shape = shape
        
        T, H, W = shape
        ct, cy, cx = T // 2, H // 2, W // 2
        tt, yy, xx = torch.meshgrid(torch.arange(T, dtype=torch.float32),
                                 torch.arange(H, dtype=torch.float32),
                                 torch.arange(W, dtype=torch.float32), indexing='ij')
        dist_sq = (tt - ct) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
        gaussian_mask = torch.exp(-dist_sq / (2.0 * sigma ** 2))
        
        self.register_buffer('gaussian_mask', gaussian_mask)
        
        if donor_magnitudes is not None and len(donor_magnitudes) > 0:
            self.donor_subject_ids = [d[0] for d in donor_magnitudes]
            donor_stack = torch.stack([d[1] for d in donor_magnitudes])
            self.register_buffer('donor_pool', donor_stack)
        else:
            self.donor_subject_ids = []
            self.register_buffer('donor_pool', torch.zeros(0))
    
    def forward(self, x):
        if not self.training: return x
        if torch.rand(1).item() > self.p: return x
        if self.donor_pool.ndim < 2 or self.donor_pool.shape[0] < 1: return x
            
        donor_idx = torch.randint(0, self.donor_pool.shape[0], (1,)).item()
        donor_mag = self.donor_pool[donor_idx]
        
        lam = torch.distributions.Beta(self.beta_alpha, self.beta_alpha).sample().item()
        orig_shape = x.shape
        frames = x.permute(1, 0, 2, 3)
        
        fft_a = torch.fft.fftshift(torch.fft.fftn(frames, dim=(1, 2, 3)), dim=(1, 2, 3))
        mag_a = torch.abs(fft_a)
        phase_a = torch.angle(fft_a)
        
        mag_donor = donor_mag.unsqueeze(0)
        mask = self.gaussian_mask.unsqueeze(0)
        blend_mask = mask * lam
        
        mixed_mag = (blend_mask * mag_donor) + ((1.0 - blend_mask) * mag_a)
        mixed_fft = mixed_mag * torch.exp(1j * phase_a)
        mixed_frames = torch.fft.ifftn(torch.fft.ifftshift(mixed_fft, dim=(1, 2, 3)), dim=(1, 2, 3)).real
        
        return mixed_frames.permute(1, 0, 2, 3).reshape(orig_shape)

def build_fourier_mixup(dataset, retention_ratio=0.95, p=0.5, beta_alpha=1.0, 
                        max_samples_per_subject=200, device='cuda',
                        cache_dir=None, loso_id=None):
    import os
    if cache_dir is None: cache_dir = getattr(dataset, 'samples_dir', None) or '.'
    cache_key = f"loso{loso_id}_ret{retention_ratio:.2f}" if loso_id is not None else f"ret{retention_ratio:.2f}"
    cache_path = os.path.join(cache_dir, f"fourier_mixup_cache_3d_{cache_key}.pt")
    
    if os.path.exists(cache_path):
        print(f"[FourierMixup] Loading cached σ from {cache_path}")
        cached = torch.load(cache_path, weights_only=False)
        sigma = cached['sigma'].item()
        shape = tuple(cached['shape'].tolist())
        donor_ids = cached.get('donor_subject_ids', [])
        donor_centroids = cached.get('donor_centroids', None)
        if donor_centroids is not None and len(donor_ids) > 0:
            donor_magnitudes = list(zip(donor_ids, [donor_centroids[i] for i in range(donor_centroids.shape[0])]))
        else:
            donor_magnitudes = []
    else:
        print(f"[FourierMixup] No cache found at {cache_path}, computing from scratch...")
        variance_map, shape, donor_magnitudes = compute_subject_variance_map(
            dataset, max_samples_per_subject=max_samples_per_subject, device=device
        )
        sigma = compute_optimal_sigma(variance_map, retention_ratio=retention_ratio)
        try:
            donor_ids = [d[0] for d in donor_magnitudes]
            donor_centroids = torch.stack([d[1] for d in donor_magnitudes]) if donor_magnitudes else torch.zeros(0)
            torch.save({
                'sigma': torch.tensor(sigma),
                'shape': torch.tensor(shape),
                'variance_map': variance_map,
                'retention_ratio': torch.tensor(retention_ratio),
                'donor_subject_ids': donor_ids,
                'donor_centroids': donor_centroids,
            }, cache_path)
        except Exception as e:
            print(f"[FourierMixup] Warning: could not save cache: {e}")
    
    mixup = FourierMixup(sigma=sigma, shape=shape, p=p, beta_alpha=beta_alpha, donor_magnitudes=donor_magnitudes)
    print(f"[FourierMixup] Ready: σ={sigma:.2f}, shape={shape}, p={p}, β_α={beta_alpha}")
    return mixup
