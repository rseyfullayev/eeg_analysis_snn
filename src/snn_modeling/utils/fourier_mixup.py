"""
Fourier Domain Adaptation Mixup (Soft FDA) for EEG Topomaps.

Instead of adversarial training (DANN/MMD) at the loss level, this module
performs identity erasure at the **data level** by smoothly blending the
low-frequency (subject-specific) magnitude spectrum between subjects while
preserving the high-frequency (emotion-specific) content.

Three-phase pipeline:
  Phase 1 — Offline: Compute a 2D Subject Variance Map from training FFTs.
  Phase 2 — Binary Search: Find the optimal Gaussian σ that retains a target
             fraction of the subject variance energy (the "perplexity trick").
  Phase 3 — Online: Apply soft FDA mixup per batch during training.

Reference:
  Yang & Soatto, "FDA: Fourier Domain Adaptation for Semantic Segmentation", CVPR 2020.
  Adapted for EEG topographic maps with subject-variance-driven sigma selection.
"""

import torch
import torch.nn as nn
import torch.fft
import numpy as np
from tqdm import tqdm


# ======================== Phase 1: Offline Variance Map ======================== #

def compute_subject_variance_map(dataset, max_samples_per_subject=200, device='cuda'):
    """Compute the 2D inter-subject variance map from training FFT magnitudes.
    
    Args:
        dataset: SWEEPDataset (train split). Each sample returns a bag_video 
                 tensor of shape [K, T, C, H, W] or [T, C, H, W].
        max_samples_per_subject: cap per subject to keep compute bounded.
        device: 'cuda' or 'cpu'.
    
    Returns:
        variance_map: (H, W) float tensor — bright = high subject bias.
        grid_size: int — spatial dimension H=W.
    """
    from collections import defaultdict
    
    subject_magnitudes = defaultdict(list)  # subject_id -> list of mean magnitude spectra
    
    print("[FourierMixup] Phase 1: Computing subject magnitude centroids...")
    
    for idx in tqdm(range(len(dataset)), desc="Scanning FFTs"):
        sample = dataset[idx]
        # Handle augmented (8-item) vs non-augmented (6-item) returns
        if len(sample) == 8:
            video = sample[0]  # inp1
            subject_idx = sample[4]
        else:
            video = sample[0]  # bag_video
            subject_idx = sample[2] if len(sample) >= 4 else -1
            # Actually for non-augmented: (bag_video, target_map, label_idx, bag_id, video_id, timestamp)
            # subject_idx is stored in dataset.samples
            subject_idx = dataset.samples[idx][3]
            
        if subject_idx == -1:
            continue
            
        # Already have enough for this subject?
        if len(subject_magnitudes[subject_idx]) >= max_samples_per_subject:
            continue
            
        # video shape: [K, T, C, H, W] (bagged) or [T, C, H, W] (non-bagged)
        if video.dim() == 5:
            # Take first bag element, average over time and channels
            frame = video[0].mean(dim=[0, 1])  # (H, W)
        elif video.dim() == 4:
            frame = video.mean(dim=[0, 1])  # (H, W)
        else:
            continue
            
        frame = frame.to(device).float()
        
        # 2D FFT + center shift
        fft_2d = torch.fft.fft2(frame)
        fft_shifted = torch.fft.fftshift(fft_2d)
        magnitude = torch.abs(fft_shifted)
        
        subject_magnitudes[subject_idx].append(magnitude.cpu())
    
    # Compute per-subject centroids (mean magnitude spectrum)
    subject_ids_sorted = sorted(subject_magnitudes.keys())
    centroids = []
    donor_magnitudes = []  # (subject_id, centroid) pairs for donor pool
    for subj_id in subject_ids_sorted:
        mags = torch.stack(subject_magnitudes[subj_id])  # (N_samples, H, W)
        centroid = mags.mean(dim=0)  # (H, W)
        centroids.append(centroid)
        donor_magnitudes.append((subj_id, centroid))
    
    if len(centroids) < 2:
        print("[FourierMixup] Warning: fewer than 2 subjects found, returning uniform variance map.")
        H = centroids[0].shape[0] if centroids else 32
        return torch.ones(H, H), H, donor_magnitudes
        
    centroids = torch.stack(centroids)  # (N_subjects, H, W)
    
    # Inter-subject variance at each frequency coordinate
    variance_map = centroids.var(dim=0)  # (H, W)
    
    # Normalize to [0, 1] for interpretability
    variance_map = variance_map / (variance_map.max() + 1e-8)
    
    grid_size = variance_map.shape[0]
    print(f"[FourierMixup] Phase 1 complete: {len(centroids)} subjects, grid={grid_size}x{grid_size}")
    
    return variance_map, grid_size, donor_magnitudes


# ======================== Phase 2: Binary Search for σ ======================== #

def compute_optimal_sigma(variance_map, retention_ratio=0.95, max_iter=50, tol=1e-6):
    """Binary search for the Gaussian σ that retains `retention_ratio` of subject variance energy.
    
    Args:
        variance_map: (H, W) tensor — the subject variance map.
        retention_ratio: fraction of total energy to retain (e.g. 0.95).
        max_iter: max binary search iterations.
        tol: convergence tolerance on energy.
    
    Returns:
        sigma: float — optimal Gaussian bandwidth.
    """
    H, W = variance_map.shape
    total_energy = variance_map.sum().item()
    target_energy = retention_ratio * total_energy
    
    # Create coordinate grid centered at DC
    cy, cx = H // 2, W // 2
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                             torch.arange(W, dtype=torch.float32), indexing='ij')
    dist_sq = (yy - cy) ** 2 + (xx - cx) ** 2
    
    sigma_min = 0.5
    sigma_max = float(max(H, W))
    
    print(f"[FourierMixup] Phase 2: Binary search for σ (target retention={retention_ratio:.2f})...")
    
    for i in range(max_iter):
        sigma = (sigma_min + sigma_max) / 2.0
        gaussian = torch.exp(-dist_sq / (2.0 * sigma ** 2))
        filtered_energy = (variance_map * gaussian).sum().item()
        
        if abs(filtered_energy - target_energy) / (total_energy + 1e-8) < tol:
            break
            
        if filtered_energy < target_energy:
            sigma_min = sigma  # filter too narrow, widen
        else:
            sigma_max = sigma  # filter too wide, narrow
    
    print(f"[FourierMixup] Phase 2 complete: optimal σ = {sigma:.2f} "
          f"(retained {filtered_energy/total_energy:.4f} of variance energy)")
    
    return sigma


# ======================== Phase 3: Online Soft FDA Mixup ======================== #

class FourierMixup(nn.Module):
    """Online Soft FDA Mixup augmentation for EEG topomaps.
    
    Blends the low-frequency (subject-biased) magnitude spectrum of
    one sample with that of a random donor subject, while keeping the
    high-frequency (emotion) content and phase intact.
    
    Works at per-sample level: each call fetches a random donor from a
    precomputed pool of per-subject magnitude centroids. This enables
    inter-subject mixing even inside __getitem__.
    
    Args:
        sigma: Gaussian bandwidth (from Phase 2 binary search).
        grid_size: spatial size H=W of topomaps.
        p: probability of applying the mixup.
        beta_alpha: Beta distribution alpha for blending strength.
        donor_magnitudes: list of (subject_id, magnitude_centroid) tuples.
            Each magnitude_centroid is a (H, W) tensor — the mean FFT
            magnitude for that subject. Built during Phase 1.
    """
    
    def __init__(self, sigma, grid_size, p=0.5, beta_alpha=1.0, donor_magnitudes=None):
        super().__init__()
        self.p = p
        self.beta_alpha = beta_alpha
        self.sigma = sigma
        
        # Precompute the fixed Gaussian mask: 1.0 at DC, 0.0 at edges
        cy, cx = grid_size // 2, grid_size // 2
        yy, xx = torch.meshgrid(torch.arange(grid_size, dtype=torch.float32),
                                 torch.arange(grid_size, dtype=torch.float32), indexing='ij')
        dist_sq = (yy - cy) ** 2 + (xx - cx) ** 2
        gaussian_mask = torch.exp(-dist_sq / (2.0 * sigma ** 2))
        
        # Register as buffer so it moves with .to(device) automatically
        self.register_buffer('gaussian_mask', gaussian_mask)  # (H, W)
        
        # Donor pool: stack all subject centroids into a single tensor
        # Shape: (N_subjects, H, W)
        if donor_magnitudes is not None and len(donor_magnitudes) > 0:
            self.donor_subject_ids = [d[0] for d in donor_magnitudes]
            donor_stack = torch.stack([d[1] for d in donor_magnitudes])
            self.register_buffer('donor_pool', donor_stack)  # (N_subjects, H, W)
        else:
            self.donor_subject_ids = []
            self.register_buffer('donor_pool', torch.zeros(0))
    
    def forward(self, x):
        """Apply soft FDA mixup with a random donor subject.
        
        Args:
            x: Input tensor of shape [T, C, H, W] (single sample, no bagging).
            
        Returns:
            Augmented tensor of same shape.
        """
        if not self.training:
            return x
            
        if torch.rand(1).item() > self.p:
            return x
        
        if self.donor_pool.ndim < 2 or self.donor_pool.shape[0] < 1:
            return x  # No donors available
            
        # x shape: [T, C, H, W]
        H, W = x.shape[-2:]
        
        # Pick a random donor
        donor_idx = torch.randint(0, self.donor_pool.shape[0], (1,)).item()
        donor_mag = self.donor_pool[donor_idx]  # (H, W)
        
        # Blend strength from Beta distribution
        lam = torch.distributions.Beta(self.beta_alpha, self.beta_alpha).sample().item()
        
        # Flatten to (T*C, H, W) for efficient batched FFT
        orig_shape = x.shape
        frames = x.reshape(-1, H, W)
        
        # 2D FFT + center shift
        fft_a = torch.fft.fftshift(torch.fft.fft2(frames))
        
        mag_a = torch.abs(fft_a)
        phase_a = torch.angle(fft_a)
        
        # Expand donor magnitude to match frame count: (1, H, W)
        mag_donor = donor_mag.unsqueeze(0)
        
        # Gaussian blending: low-freq gets donor, high-freq keeps original
        mask = self.gaussian_mask.unsqueeze(0)  # (1, H, W)
        blend_mask = mask * lam  # Scale by lambda for soft blending
        
        mixed_mag = (blend_mask * mag_donor) + ((1.0 - blend_mask) * mag_a)
        
        # Reconstruct: mixed magnitude + original phase
        mixed_fft = mixed_mag * torch.exp(1j * phase_a)
        
        # Inverse FFT
        mixed_frames = torch.fft.ifft2(torch.fft.ifftshift(mixed_fft)).real
        
        return mixed_frames.reshape(orig_shape)


def build_fourier_mixup(dataset, retention_ratio=0.95, p=0.5, beta_alpha=1.0, 
                        max_samples_per_subject=200, device='cuda',
                        cache_dir=None, loso_id=None):
    """Full pipeline: compute variance map → binary search σ → return FourierMixup module.
    
    Results are cached to disk so Phase 1+2 only run once per LOSO fold.
    Subsequent training runs load the cached sigma + grid_size instantly.
    
    Args:
        dataset: SWEEPDataset (train split).
        retention_ratio: energy retention for σ selection (default 0.95).
        p: probability of applying mixup per sample.
        beta_alpha: Beta distribution parameter for blend strength.
        max_samples_per_subject: cap for Phase 1 scanning.
        device: compute device.
        cache_dir: directory to save/load cache (defaults to dataset_path).
        loso_id: leave-one-subject-out ID for cache key differentiation.
    
    Returns:
        FourierMixup module (nn.Module), ready to be composed with other augmentations.
    """
    import os
    
    # Build cache path — unique per LOSO fold and retention ratio
    if cache_dir is None:
        cache_dir = getattr(dataset, 'samples_dir', None) or '.'
    cache_key = f"loso{loso_id}_ret{retention_ratio:.2f}" if loso_id is not None else f"ret{retention_ratio:.2f}"
    cache_path = os.path.join(cache_dir, f"fourier_mixup_cache_{cache_key}.pt")
    
    if os.path.exists(cache_path):
        print(f"[FourierMixup] Loading cached σ from {cache_path}")
        cached = torch.load(cache_path, weights_only=False)
        sigma = cached['sigma'].item()
        grid_size = int(cached['grid_size'].item())
        # Restore donor pool from cache
        donor_ids = cached.get('donor_subject_ids', [])
        donor_centroids = cached.get('donor_centroids', None)
        if donor_centroids is not None and len(donor_ids) > 0:
            donor_magnitudes = list(zip(donor_ids, [donor_centroids[i] for i in range(donor_centroids.shape[0])]))
        else:
            donor_magnitudes = []
        print(f"[FourierMixup] Cached: σ={sigma:.2f}, grid={grid_size}, {len(donor_magnitudes)} donors")
    else:
        print(f"[FourierMixup] No cache found at {cache_path}, computing from scratch...")
        variance_map, grid_size, donor_magnitudes = compute_subject_variance_map(
            dataset, max_samples_per_subject=max_samples_per_subject, device=device
        )
        
        sigma = compute_optimal_sigma(variance_map, retention_ratio=retention_ratio)
        
        # Save to disk for future runs (including donor centroids)
        try:
            donor_ids = [d[0] for d in donor_magnitudes]
            donor_centroids = torch.stack([d[1] for d in donor_magnitudes]) if donor_magnitudes else torch.zeros(0)
            torch.save({
                'sigma': torch.tensor(sigma),
                'grid_size': torch.tensor(grid_size),
                'variance_map': variance_map,
                'retention_ratio': torch.tensor(retention_ratio),
                'donor_subject_ids': donor_ids,
                'donor_centroids': donor_centroids,
            }, cache_path)
            print(f"[FourierMixup] Cached results to {cache_path}")
        except Exception as e:
            print(f"[FourierMixup] Warning: could not save cache: {e}")
    
    mixup = FourierMixup(sigma=sigma, grid_size=grid_size, p=p, beta_alpha=beta_alpha,
                         donor_magnitudes=donor_magnitudes)
    
    print(f"[FourierMixup] Ready: σ={sigma:.2f}, grid={grid_size}, p={p}, β_α={beta_alpha}, donors={len(donor_magnitudes)}")
    
    return mixup
