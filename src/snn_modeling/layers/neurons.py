import torch
import torch.nn as nn
class LearnableAtan(nn.Module):
    """
        A PyTorch-native Learnable Arctan Surrogate.
        
        Gradient estimation for spiking neurons using a learnable slope parameter.
        
        References:
        -----------
        Adapted from 'snntorch' implementation of PLIF strategies:
        
        * W. Fang, Z. Yu, Y. Chen, T. Masquelier, T. Huang, Y. Tian (2021) 
        "Incorporating Learnable Membrane Time Constants to Enhance Learning 
        of Spiking Neural Networks." Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV).
        
        * Jason K. Eshraghian et al. (2021) "Training Spiking Neural Networks 
        Using Lessons From Deep Learning." arXiv:2109.12894 (snntorch)
    """
   
    def __init__(self, alpha=2.0, learnable=True):
        super().__init__()
        self.learnable = learnable
        if learnable:
            self.alpha = nn.Parameter(torch.tensor(float(alpha)), requires_grad=learnable)
        else:
            self.register_buffer("alpha_fixed", torch.tensor(alpha))
        
    @staticmethod
    class _Function(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            ctx.save_for_backward(input, alpha)
            return (input > 0).float()

        @staticmethod
        def backward(ctx, grad_output):
            input, alpha = ctx.saved_tensors
            slope = alpha.abs() 
            denom = 1 + (slope * input).pow(2)
            d_input = grad_output * (slope / denom)
            d_alpha = grad_output * (input / denom)
            
            return d_input, d_alpha.sum()

    def forward(self, x):
        alpha = self.alpha if self.learnable else self.alpha_fixed
        return self._Function.apply(x, alpha)

class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        if x.dim() == 5:
            T, B, C, H, W = x.shape
            x_reshaped = x.reshape(T * B, C, H, W)
            y = self.module(x_reshaped)
            _, C_out, H_out, W_out = y.shape
            y = y.reshape(T, B, C_out, H_out, W_out)
            return y
        else:
            return self.module(x)
    
    def __name__(self):
        return self.module.__name__


class ALIF(nn.Module):
    """
    Adaptive LIF with Membrane Potential Batch Normalization.
    
    - Integrates inputs over time (Leaky).
    - Normalizes Input (BN) to prevent explosion.
    - Adapts threshold (ALIF) to prevent saturation.
    - Uses HARD RESET.
    """
    def __init__(self, num_channels, beta=0.9, threshold=1.0, 
                 decay_adapt=0.96, gamma_adapt=0.5, batch_norm=True,
                 spike_grad=None, return_mem=False,
                 learn_beta=False, learn_threshold=False, 
                 learn_decay=False, learn_gamma=False,
                 learn_slope=False):
        
        super(ALIF, self).__init__()
        self.learn_beta = learn_beta
        self.learn_threshold = learn_threshold
        self.learn_decay = learn_decay
        self.learn_gamma = learn_gamma

        if learn_beta:
            self.beta_param = nn.Parameter(torch.tensor(beta).logit())
        else:
            self.register_buffer("beta_fixed", torch.tensor(beta))

        if learn_threshold:
            self.threshold_param = nn.Parameter(torch.tensor(threshold))
        else:
            self.register_buffer("threshold_fixed", torch.tensor(threshold))

        if learn_decay:
            self.decay_param = nn.Parameter(torch.tensor(decay_adapt).logit())
        else:
            self.register_buffer("decay_fixed", torch.tensor(decay_adapt))

        if learn_gamma:
            self.gamma_param = nn.Parameter(torch.tensor(gamma_adapt))
        else:
            self.register_buffer("gamma_fixed", torch.tensor(gamma_adapt))

        self.bn = nn.BatchNorm3d(num_channels, eps=1e-4) if batch_norm else nn.Identity()
        self.return_mem = return_mem
        self.spike_grad = LearnableAtan(alpha=2.0, learnable=learn_slope)
    
    def forward(self, x):
        T, B, C, H, W = x.shape

        beta = torch.sigmoid(self.beta_param) if self.learn_beta else self.beta_fixed
        threshold_base = self.threshold_param.abs() if self.learn_threshold else self.threshold_fixed
        decay_adapt = torch.sigmoid(self.decay_param) if self.learn_decay else self.decay_fixed
        gamma_adapt = self.gamma_param.abs() if self.learn_gamma else self.gamma_fixed

        mem = torch.zeros(B, C, H, W, device=x.device)
        adapt_thresh = torch.zeros(B, C, H, W, device=x.device)

        spikes = []
        mems = []

        x = x.permute(1, 2, 0, 3, 4)
        x = self.bn(x)
        x = x.permute(2, 0, 1, 3, 4)

        for t in range(T):
            # Leaky Integration
            # Standart Accumulation: mem[t] = beta * mem[t-1] + x[t]
            mem = beta * mem + x[t]
            if self.return_mem:
                mems.append(mem)
            
            else:
                # Adaptive Threshold
                effective_thresh = threshold_base + adapt_thresh

                # Spike Generation
                spike = self.spike_grad(mem - effective_thresh)
                spikes.append(spike)

                # Membrane Potential Reset (HARD RESET)
                mem = mem * (1 - spike)
                
                # Update Adaptive Threshold
                # If spike occurs, increase threshold. Decay over time.
                adapt_thresh = (decay_adapt * adapt_thresh) + (gamma_adapt * spike)

        self.mem = mem.detach() 

        if self.return_mem:
            return torch.stack(mems, dim=0)
        else:
            return torch.stack(spikes, dim=0)