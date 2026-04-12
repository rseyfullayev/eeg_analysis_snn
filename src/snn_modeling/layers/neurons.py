import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd


'''
@inproceedings{li2022odconv,
  title={Omni-Dimensional Dynamic Convolution},
  author={Chao Li and Aojun Zhou and Anbang Yao},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=DmpCfq6Mg39}
}

'''

class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16, use_batchnorm=False):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel) if use_batchnorm else nn.GroupNorm(1, attention_channel, affine=True)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)

'''
@inproceedings{li2022odconv,
  title={Omni-Dimensional Dynamic Convolution},
  author={Chao Li and Aojun Zhou and Anbang Yao},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=DmpCfq6Mg39}
}

'''

class ODConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=4, use_batchnorm=False):
        super(ODConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups,
                                   reduction=reduction, kernel_num=kernel_num, use_batchnorm=use_batchnorm)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, kernel_size, kernel_size),
                                   requires_grad=True)
        self._initialize_weights()

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x):
        # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
        # while we observe that when using the latter method the models will run faster with less gpu memory cost.
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention
        return output

    def _forward_impl_pw1x(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        return self._forward_impl(x)



class TemporalShift(nn.Module):
    def __init__(self, n_div=8):
        super(TemporalShift, self).__init__()
        self.fold_div = n_div

    def forward(self, x):
        # x is [B*T, C, H, W]
        T, B, C, H , W = x.size()
        fold = C // self.fold_div
        
        out = torch.zeros_like(x)
        
        # 1. Bidirectional Shift
        # Shift Left (Future -> Present)
        out[:-1, :, :fold] = x[1:, :, :fold] 
        # Shift Right (Past -> Present)
        out[1:, :, fold:2*fold] = x[:-1, :, fold:2*fold]
        # Center (Static)
        out[:, :, 2*fold:] = x[:, :, 2*fold:]

        return out

class SwiGLU(nn.Module):
    """
    SwiGLU Activation Function.
    
    Combines SiLU (Swish) activation with Gated Linear Units (GLU).
    
    References:
    -----------
    * "SwiGLU: A New Activation Function for Language Models" 
      by Shazeer et al., 2020. https://arxiv.org/abs/2002.05202
    """
    def __init__(self, in_features, p_drop=0.1):
        super(SwiGLU, self).__init__()
        self.gate = nn.Linear(in_features, in_features*4, bias=False)
        self.value = nn.Linear(in_features, in_features*4, bias=False)
        self.out = nn.Linear(in_features*4, in_features, bias=False)
        self.silu = nn.SiLU()
        self.drop = nn.Dropout(p_drop)

        
    def forward(self, x):
        x_gate = self.gate(x)
        x_val = self.value(x)
        x = self.silu(x_gate) * x_val
        return self.drop(self.out(x))

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

class TemporalOrderFix(nn.Module):
    def __init__(self, module):
        super(TemporalOrderFix, self).__init__()
        self.module = module

    def forward(self, x):
        if x.dim() == 5:
            x_permuted = x.permute(1, 2, 0, 3, 4)#.contiguous()  # B, C, T, H, W
            x_permuted = self.module(x_permuted)
            x_permuted = x_permuted.permute(2, 0, 1, 3, 4)#.contiguous()  # T, B, C, H, W
            return x_permuted
        else:
            return self.module(x)
    
    def __name__(self):
        return self.module.__name__


class ALIF(nn.Module):
    """
    Adaptive LIF with Batch Normalization.
    
    - Integrates inputs over time (Leaky).
    - Normalizes Input (BN) to prevent explosion.
    - Adapts threshold (ALIF) to prevent saturation.
    - Uses HARD RESET.
    """
    def __init__(self, num_channels, beta=0.9, threshold=1.0, 
                 decay_adapt=0.96, gamma_adapt=0.5, batch_norm=True,
                 norm_mem = nn.InstanceNorm3d,
                 spike_grad=None, return_mem=False,
                 learn_beta=False, learn_threshold=False, 
                 learn_decay=False, learn_gamma=False,
                 learn_slope=False, recurrent=False, kernel_size=3):
        
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

        self.bn = TemporalOrderFix(nn.BatchNorm3d(num_channels, eps=1e-4)) if batch_norm else nn.Identity()
        if return_mem:
            self.mem_ = nn.Sequential(TemporalOrderFix(norm_mem(num_channels, eps=1e-4)), 
                                      nn.SiLU())
        self.return_mem = return_mem
        self.spike_grad = LearnableAtan(alpha=2.0, learnable=learn_slope)

        self.recurrent = recurrent
        if recurrent:
            padding = kernel_size // 2
            self.recurrent_conv = nn.utils.spectral_norm(nn.Conv2d(num_channels, num_channels, kernel_size=kernel_size, padding=padding, bias=False))

    
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
        spike_prev = torch.zeros(B, C, H, W, device=x.device)

        x = self.bn(x)

        for t in range(T):
            # Recurrent Contribution
            input_t = x[t]
            if self.recurrent:
                input_t += self.recurrent_conv(spike_prev)
            # Leaky Integration
            # Standart Accumulation: mem[t] = beta * mem[t-1] + input_t
            mem = beta * mem + input_t
            if self.return_mem:
                mems.append(mem)
            
            else:
                # Adaptive Threshold
                effective_thresh = threshold_base + adapt_thresh

                # Spike Generation
                spike = self.spike_grad(mem - effective_thresh)
                spikes.append(spike)

                # Membrane Potential Reset (SOFT RESET)
                # mem = mem * (1 - spike)
                mem = mem - (spike * effective_thresh)
                
                # Update Adaptive Threshold
                # If spike occurs, increase threshold. Decay over time.
                adapt_thresh = (decay_adapt * adapt_thresh) + (gamma_adapt * spike)

                spike_prev = spike

        # Detach final membrane state to prevent graph retention
        self.mem = mem.detach()

        if self.return_mem:
            # Stack and return; clear local list
            result = self.mem_(torch.stack(mems, dim=0))
            mems.clear()
            return result
        else:
            result = torch.stack(spikes, dim=0)
            spikes.clear()
            return result