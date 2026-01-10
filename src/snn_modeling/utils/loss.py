import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import TverskyLoss
import torch.nn.functional as F

class FiringRateRegularizer:
    def __init__(self, model, target_rate=0.05, lambda_reg=0.1):

        self.target_rate = target_rate
        self.lambda_reg = lambda_reg
        self.layer_outputs = {}
        self.hooks = []
        
        self._register_hooks(model)

    def _register_hooks(self, model):
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, torch.Tensor):
                    # Detach to prevent holding computation graph in memory
                    self.layer_outputs[name] = output.detach()
            return hook

        for name, layer in model.named_modules():
            if "ALIF" in str(type(layer)): 
                if hasattr(layer, 'return_mem') and layer.return_mem:
                    continue 
                self.hooks.append(layer.register_forward_hook(get_activation(name)))

    def compute_tax(self):
        reg_loss = 0
        for _, spikes in self.layer_outputs.items():
            firing_rate = torch.mean(spikes) 
            reg_loss += (firing_rate - self.target_rate) ** 2

        # Clear stored outputs to free memory
        self.layer_outputs.clear()
        
        return self.lambda_reg * reg_loss

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


class TopKClassificationLoss(nn.Module):
    def __init__(self, k_percent=0.05):
        super(TopKClassificationLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.k_percent = k_percent
        self.scale = nn.Parameter(torch.tensor(5.0))

    def forward(self, inputs, targets_class):
        B, C, H, W = inputs.shape
        flat_inputs = inputs.view(B, C, -1)
        
        k = max(1, int(H * W * self.k_percent))

        top_k_values, _ = torch.topk(flat_inputs, k, dim=2)
        peak_logits = torch.mean(top_k_values, dim=2)
        safe_scale = F.softplus(self.scale)
        peak_logits_scaled = peak_logits * safe_scale

        loss = self.cross_entropy(peak_logits_scaled, targets_class)

        return loss

class ContrastiveLoss(nn.Module):

    """
    Reference:
    @misc{kim2025temperaturefree,
        title={Temperature-Free Loss Function for Contrastive Learning}, 
        author={Bum Jun Kim and Sang Woo Kim},
        year={2025},
        eprint={2501.17683},
        archivePrefix={arXiv},
        primaryClass={cs.LG},
        url={https://arxiv.org/abs/2501.17683}
    }
    Implementation Note:
    Replaces standard exp(sim / temp) with exp(arctanh(sim)) to prevent 
    gradient vanishing on well-clustered embeddings.
    """
    
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, features, labels):
        # features: (N, D) where N is batch size and D is feature dimension. Normalized.
        # labels: (N,) with integer class labels

        device = features.device
        
        # 2. Cosine Similarity
        similarity_matrix = torch.matmul(features, features.T)
        
        # 3. Arctanh warping
        # Clamp to avoid infinity at exactly 1.0 or -1.0
        # arctanh(x) = 0.5 * log((1+x)/(1-x))

        eps = 1e-6
        sim_clamped = torch.clamp(similarity_matrix, -1 + eps, 1 - eps)
        logits = 0.5 * torch.log((1 + sim_clamped) / (1 - sim_clamped))
        
        
        # Create Mask
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Remove self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask), 
            1, 
            torch.arange(mask.shape[0]).view(-1, 1).to(device), 
            0
        )
        mask = mask * logits_mask
        
        # Numerical Stability for LogSumExp
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # Mean Log-Likelihood
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        
        loss = - mean_log_prob_pos.mean()
        return loss


class FullHybridLoss(nn.Module):
    def __init__(self,
                 time_steps=16,
                 smooth=0., 
                 lambda_seg=1., 
                 lambda_con=1., 
                 lambda_class=0.1, 
                 alpha=0.5, 
                 beta=0.5):
        super().__init__()

        self.con_loss = ContrastiveLoss()
        self.dice_loss = TverskyLoss(mode="multilabel", smooth=smooth, from_logits=True, alpha=alpha, beta=beta)
        self.class_loss = TopKClassificationLoss()
        self.lambda_class = lambda_class
        self.lambda_seg = lambda_seg
        self.lambda_con = lambda_con
        self.time_steps = time_steps
    
    def add_fire_rate_loss(self, model, lambda_fire=0.1, target_rate=0.05):
        self.fire_loss = FiringRateRegularizer(model, target_rate=target_rate, lambda_reg=lambda_fire)

    def forward(self, inputs, targets_mask, targets_class):
        segmentation_loss, classification_loss, con_loss = 0, 0, 0
        if type(inputs) is tuple:
            inputs, embedding = inputs

        if self.lambda_seg > 0:
            segmentation_loss = self.dice_loss(inputs, targets_mask)
        if self.lambda_class > 0:
            classification_loss = self.class_loss(inputs, targets_class)
        if self.lambda_con > 0:
            con_loss = self.con_loss(embedding, targets_class.repeat_interleave(self.time_steps))
        

        total_loss = self.lambda_seg * segmentation_loss + self.lambda_class * classification_loss + self.lambda_con * con_loss
        
        if hasattr(self, 'fire_loss'):
            tax_loss = self.fire_loss.compute_tax()
            total_loss += tax_loss
        
        return total_loss
