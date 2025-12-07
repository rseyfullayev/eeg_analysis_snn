import torch
import torch.nn as nn
import numpy as np
import pywt
import scipy.signal

class WaveletModule(nn.Module):
    class WaveletModule(nn.Module):
        def __init__(self, window_size=256, target_steps=32, fs=200):
            super().__init__()
            self.window_size = window_size
            self.target_steps = target_steps
            self.fs = fs
            self.wavelet_name = "cmor1.5-1.0"

            self.bands = {
                "delta": (1, 4),
                "theta": (4, 8),
                "alpha": (8, 14),
                "beta": (14, 31),
                "gamma": (31, 50)
            }

    def create_windows(self, eeg, step_size=128):
        windows = []
        channels, total_samples = eeg.shape

        if total_samples < self.window_size:
            return torch.empty(0)
        
        assert channels == 62

        for start in range(0, total_samples - self.window_size + 1, step_size):
            end = start + self.window_size
            window = eeg[:, start:end]
            windows.append(window)

        if len(windows) == 0: return torch.empty(0)

        return torch.stack(windows)

    def wavelet(self, window):
        if isinstance(window, torch.Tensor):
            data = window.cpu().numpy()
        else:
            data = window
        n_channels, n_time = data.shape
        self.freqs = np.arange(1, 51, 0.5) 
        scales = pywt.frequency2scale(self.wavelet_name, self.freqs) * self.fs
        coeffs = []

        for channel in range(n_channels):
            cwtmatr, _ = pywt.cwt(
                data[channel],
                scales,
                self.wavelet_name,
                sampling_period=1 / self.fs
            )
            power = np.abs(cwtmatr) ** 2
            power_tensor = torch.tensor(power, dtype=torch.float32) 
            coeffs.append(power_tensor)

        return torch.stack(coeffs)
    
    def bandpowers_wavelet(self, coeffs):
        band_powers = []
        freqs = torch.tensor(self.freqs, dtype=torch.float32)
        for band_name, (low, high) in self.bands.items():
            idx = (freqs >= low) & (freqs <= high)
            if idx.sum() > 0:
                power = coeffs[:, idx, :].mean(dim=1)
            else:
                power = torch.zeros(coeffs.shape[0], coeffs.shape[2])
            
            band_powers.append(power)

        return torch.stack(band_powers, dim=1)
    
    def resample_time(self, features):
        feat_np = features.cpu().numpy()
        resampled_np = scipy.signal.resample(feat_np, self.target_steps, axis=-1)
        return torch.tensor(resampled_np, dtype=torch.float32)
