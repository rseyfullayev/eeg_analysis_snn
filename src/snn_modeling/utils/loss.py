import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import TverskyLoss


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
                if isinstance(output, torch.Tensor) and output.requires_grad:
                     self.layer_outputs[name] = output
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

        self.layer_outputs = {}
        
        return self.lambda_reg * reg_loss

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


class TopKClassificationLoss(nn.Module):
    def __init__(self, k_percent=0.1):
        super(TopKClassificationLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.k_percent = k_percent
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, inputs, targets_class):
        B, C, H, W = inputs.shape
        flat_inputs = inputs.view(B, C, -1)
        
        k = max(1, int(H * W * self.k_percent))
 
        top_k_values, _ = torch.topk(flat_inputs, k, dim=2)
        peak_logits = torch.mean(top_k_values, dim=2) 
        peak_logits = peak_logits * self.scale
        
        return self.cross_entropy(peak_logits, targets_class)


class FullHybridLoss(nn.Module):
    def __init__(self, smooth=0., lambda_seg=1., lambda_bce=1., lambda_class=0.1, alpha=0.5, beta=0.5, label_smooth=0.):
        super().__init__()

        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.dice_loss = TverskyLoss(mode="multilabel", smooth=smooth, from_logits=True, alpha=alpha, beta=beta)
        self.class_loss = TopKClassificationLoss()
        self.lambda_class = lambda_class
        self.lambda_seg = lambda_seg
        self.lambda_bce = lambda_bce
        self.label_smooth = label_smooth

    def forward(self, inputs, targets_mask, targets_class):
        segmentation_loss = self.dice_loss(inputs, targets_mask)
        classification_loss = self.class_loss(inputs, targets_class)
        with torch.no_grad():
            smooth_targets = targets_mask * (1 - self.smoothing) + 0.5 * self.smoothing
        
        bce_loss = self.bce_loss(inputs, smooth_targets)

        total_loss = self.lambda_seg * segmentation_loss + self.lambda_class * classification_loss + self.lambda_bce * bce_loss
        
        
        return total_loss