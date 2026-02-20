import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class WaveletModule(nn.Module):
    def __init__(self, fs=200, target_steps=32, device='cuda'):
        super().__init__()
        self.device = device
        self.output_steps = target_steps
        
        self.freqs = np.arange(1, 51, 0.5)
        
        # Define Scales (Center Freq = 1.0, B = 3.0)
        center_freq = 1.0 
        self.scales = (center_freq * fs) / self.freqs
 
        B = 3.0
        kernels = []
        for scale in self.scales:
            sigma = scale 
            t = np.arange(-4*sigma, 4*sigma + 1) / fs
            
            norm = (np.pi * B) ** -0.25
            oscillation = np.exp(2j * np.pi * center_freq * (t * fs / scale))
            envelope = np.exp(-(t * fs / scale)**2 / B)

            wavelet = (scale ** -0.5) * norm * oscillation * envelope
            kernels.append(torch.tensor(wavelet, dtype=torch.complex64))

        max_len = max([k.shape[0] for k in kernels])

        if max_len % 2 == 0:
            max_len += 1

        self.padding = max_len // 2


        self.max_kernel_len = max_len
        
        self.weights = torch.zeros(len(self.freqs), 1, max_len, dtype=torch.complex64)
        
        for i, k in enumerate(kernels):
            pad = (max_len - k.shape[0]) // 2
            self.weights[i, 0, pad : pad + k.shape[0]] = k
            
        self.weights = self.weights.to(device)

        self.band_indices = self._make_band_masks()

    def _make_band_masks(self):
        bands = {
            "delta": (1, 4), "theta": (4, 8), "alpha": (8, 14), 
            "beta": (14, 31), "gamma": (31, 50)
        }
        indices = {}
        freqs_t = torch.tensor(self.freqs, device=self.device)
        for name, (low, high) in bands.items():
            mask = (freqs_t >= low) & (freqs_t <= high)
            if mask.sum() == 0: 
                diff = (freqs_t - (low+high)/2).abs()
                mask[diff.argmin()] = True
            indices[name] = mask
        return indices

    def forward(self, eeg_data):
        
        if eeg_data.dim() == 2:
            eeg_data = eeg_data.unsqueeze(0)

        B, C, T = eeg_data.shape
        x = eeg_data.reshape(B * C, 1, T)
        
        
        x = F.pad(x, (self.padding, self.padding), mode='reflect')

        # 1. CWT Convolution
        # result: (Batch*Channels, Freqs, Time_Padded)
        cwt_complex = F.conv1d(x.to(dtype=torch.complex64), self.weights)
        #print(cwt_complex.shape)

        # 4. Power & Band Integration
        power = cwt_complex.abs().pow(2) 
        
        band_powers = []
        for name, mask in self.band_indices.items():
            p = power[:, mask, :].mean(dim=1) 
            band_powers.append(p)
        
        out = torch.stack(band_powers, dim=1) # (B*C, 5, Time)
        
        # 6. Final Shape
        out = out.view(B, C, 5, T)
      
            
        return out