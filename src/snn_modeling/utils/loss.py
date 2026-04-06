import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import TverskyLoss, FocalLoss
import segmentation_models_pytorch as smp
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
                # DO NOT DETACH. We need the gradient history.
                if isinstance(output, torch.Tensor):
                    self.layer_outputs[name] = output
 
            return hook

        for name, layer in model.named_modules():
            if "ALIF" in str(type(layer)): 
                if hasattr(layer, 'return_mem') and layer.return_mem:
                    continue
                self.hooks.append(layer.register_forward_hook(get_activation(name)))

    def compute_tax(self):
        reg_loss = 0
        count = 0
        for _, spikes in self.layer_outputs.items():
            firing_rate = torch.mean(spikes) 
            reg_loss += (firing_rate - self.target_rate) ** 2
            count += 1

        self.layer_outputs = {} 
        
        if count == 0: return torch.tensor(0.0, device=spikes.device)
        return self.lambda_reg * (reg_loss / count)

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
    
class GSPLoss(nn.Module):
    def __init__(self, num_classes=5, device='cuda'):
        super(GSPLoss, self).__init__()
        self.num_classes = num_classes
        self._masks = None
        self.ce = nn.CrossEntropyLoss()
        self.device = device
        self.scale = nn.Parameter(torch.tensor(0.1)) 

    
    @property
    def masks(self):
        return self._masks
    
    @masks.setter
    def masks(self, value):
        self._masks = value.unsqueeze(0).to(self.device)
    
    def forward(self, inputs, targets_class):
        B, _, H, W = inputs.shape
        msk_act = inputs * self.masks
        gsp = msk_act.sum(dim=(2, 3))
        safe_scale = F.softplus(self.scale)
        logits = gsp * safe_scale
        loss = self.ce(logits, targets_class)
        return loss
    
     

class ContrastiveLoss(nn.Module):

    """
    Supervised Contrastive Loss with temperature scaling.
    
    Reference:
    @misc{khosla2020supervised,
        title={Supervised Contrastive Learning}, 
        author={Prannay Khosla et al.},
        year={2020},
        eprint={2004.11362},
        archivePrefix={arXiv},
    }
    """
    
    def __init__(self,
                 masks,
                 temperature=0.07,
                 iic_enabled=False,
                 iic_intra_weight=1.0,
                 iic_inter_weight=1.0):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.iic_enabled = iic_enabled
        self.iic_intra_weight = iic_intra_weight
        self.iic_inter_weight = iic_inter_weight
        sim_score = self.compute_mask_dice_matrix(masks)  # (C, C) precomputed similarity between prototype masks
        self.register_buffer('sim_score', sim_score)  # (C, C) buffer for efficient lookup

    def compute_mask_dice_matrix(self, prototypes, threshold=0.1):
        """Precompute pairwise Dice scores between C prototype masks using smp.

        Args:
            prototypes: (C, H, W) soft prototype masks.
            threshold: binarisation threshold (same as used in target_map).

        Returns:
            dice_matrix: (C, C) tensor of pairwise Dice scores.
        """
        print(prototypes.shape)
        C = prototypes.shape[0]
        binary = (prototypes > threshold).long()  # (C, H, W)

        # Build all C×C pairs: pred[i*C+j] = mask_i, target[i*C+j] = mask_j
        pred   = binary.unsqueeze(1).expand(C, C, -1, -1).reshape(C * C, 1, *binary.shape[1:])
        target = binary.unsqueeze(0).expand(C, C, -1, -1).reshape(C * C, 1, *binary.shape[1:])

        tp, fp, fn, tn = smp.metrics.get_stats(pred, target, mode='binary')
        f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction='none')  # (C*C,)
        dice_matrix = f1.reshape(C, C)
        return dice_matrix


    def forward(self, features, labels):
        # features: (N, 2, D) or (N, D) where N is batch size and D is feature dimension.
        # labels: (N,) with integer class labels

        device = features.device
        if features.dim() == 3:
            f1, f2 = torch.unbind(features, dim=1) 
            features = torch.cat([f1, f2], dim=0)  # (2N, D)
            labels = torch.cat([labels, labels], dim=0)  # (2N,)
        
        # Normalize features
        features = F.normalize(features, dim=1, eps=1e-6)
        
        # Cosine similarity scaled by temperature
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create positive mask (same class)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Remove self-contrast (diagonal)
        batch_size = features.shape[0]
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask
        
        # Build per-pair Dice weights from precomputed sim_score matrix
        label_flat = labels.squeeze(1)                                     # (2N,)
        dice_pair = self.sim_score[label_flat][:, label_flat]   # (2N, 2N)
        neg_mask = logits_mask - mask                                      # 1 for negatives, 0 for positives/self
        # Baseline uses Dice-weighted negatives; IIC additionally reweights both
        # intra-class positives and inter-class negatives.
        if self.iic_enabled:
            pos_weights = mask * self.iic_intra_weight
            neg_weights = neg_mask * (1.0 + self.iic_inter_weight * dice_pair)
            denom_weights = pos_weights + neg_weights
        else:
            # Negatives weighted by (1 + dice), positives by 1, self by 0
            denom_weights = logits_mask + neg_mask * dice_pair

        # Numerical Stability: subtract max for LogSumExp
        logits_max, _ = torch.max(similarity_matrix * logits_mask, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Compute log softmax with dice-scaled denominator
        exp_logits = torch.exp(logits) * denom_weights
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # Mean log-likelihood over positives
        mask_sum = mask.sum(1)
        # Avoid division by zero for samples with no positives
        mask_sum = torch.clamp(mask_sum, min=1.0)
        if self.iic_enabled:
            weighted_pos = mask * self.iic_intra_weight
            weighted_pos_sum = torch.clamp(weighted_pos.sum(1), min=1.0)
            mean_log_prob_pos = (weighted_pos * log_prob).sum(1) / weighted_pos_sum
        else:
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        
        loss = -mean_log_prob_pos.mean()
        return loss


class SupMoCoLoss(ContrastiveLoss):
    """Decoupled supervised MoCo objective with Dice-derived bounded weights.

    Decoupled means positives are excluded from the denominator partition.
    """

    def __init__(self,
                 masks,
                 temperature=0.07,
                 iic_enabled=False,
                 iic_intra_weight=1.0,
                 iic_inter_weight=1.0,
                 temporal_decay_enabled=False,
                 temporal_decay_factor=0.999):
        super().__init__(
            masks=masks,
            temperature=temperature,
            iic_enabled=iic_enabled,
            iic_intra_weight=iic_intra_weight,
            iic_inter_weight=iic_inter_weight,
        )
        self.temporal_decay_enabled = temporal_decay_enabled
        self.temporal_decay_factor = temporal_decay_factor

    def forward(self,
                query_features,
                key_features,
                labels,
                subject_labels,
                queue_features=None,
                queue_labels=None,
                queue_subject_labels=None,
                queue_ages=None):
        device = query_features.device
        q = F.normalize(query_features, dim=1, eps=1e-6)
        k = F.normalize(key_features, dim=1, eps=1e-6)
        labels = labels.contiguous().view(-1)
        subject_labels = subject_labels.contiguous().view(-1)

        candidate_features = [k]
        candidate_labels = [labels]
        candidate_subject_labels = [subject_labels]
        # Current batch negative samples have age 0
        candidate_ages = [torch.zeros_like(labels, dtype=torch.float32, device=device)]

        if queue_features is not None and queue_labels is not None and queue_features.numel() > 0:
            valid = queue_labels >= 0
            if valid.any():
                q_feat = F.normalize(queue_features[valid].to(device), dim=1, eps=1e-6)
                q_lbl = queue_labels[valid].to(device)
                candidate_features.append(q_feat)
                candidate_labels.append(q_lbl)
                if queue_subject_labels is not None:
                    q_subj = queue_subject_labels[valid].to(device)
                else:
                    q_subj = torch.full_like(q_lbl, -1)
                candidate_subject_labels.append(q_subj)
                
                if queue_ages is not None:
                    candidate_ages.append(queue_ages[valid].float().to(device))
                else:
                    candidate_ages.append(torch.zeros_like(q_lbl, dtype=torch.float32, device=device))

        all_features = torch.cat(candidate_features, dim=0)  # (M, D)
        all_labels = torch.cat(candidate_labels, dim=0)      # (M,)
        all_subject_labels = torch.cat(candidate_subject_labels, dim=0)  # (M,)
        all_ages = torch.cat(candidate_ages, dim=0)          # (M,)
        

        logits = torch.matmul(q, all_features.T) / self.temperature  # (B, M)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        pos_mask = (labels.unsqueeze(1) == all_labels.unsqueeze(0)).float()
        neg_mask = 1.0 - pos_mask
        same_subj_mask = (subject_labels.unsqueeze(1) == all_subject_labels.unsqueeze(0)).float()

        # Weight same-emotion positives differently depending on whether the subject matches.
        subj_weight = torch.where(
            same_subj_mask.bool(),
            torch.ones_like(same_subj_mask),
            torch.full_like(same_subj_mask, self.iic_intra_weight),
        )
        pos_weights = pos_mask * subj_weight

        # Reuse prototype-Dice pair weights for bounded negatives.
        dice_pair = self.sim_score[labels][:, all_labels]
        neg_weights = neg_mask * torch.clamp(dice_pair * self.iic_inter_weight, min=1e-6, max=1.0)
        
        if self.temporal_decay_enabled:
            # Apply temporal decay: w_temporal = temporal_decay_factor ^ age
            temporal_weights = self.temporal_decay_factor ** all_ages  # (M,)
            neg_weights = neg_weights * temporal_weights.unsqueeze(0)

        # Decoupled denominator: negatives only (no positive terms in partition).
        neg_partition = torch.clamp((torch.exp(logits) * neg_weights).sum(dim=1, keepdim=True), min=1e-8)
        pos_log_prob = logits - torch.log(neg_partition)

        pos_weight_sum = torch.clamp(pos_weights.sum(dim=1), min=1e-8)
        mean_log_prob_pos = (pos_weights * pos_log_prob).sum(dim=1) / pos_weight_sum
        return -mean_log_prob_pos.mean()


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

        self.con_loss =  nn.BCEWithLogitsLoss() #ContrastiveLoss()
        self.dice_loss = TverskyLoss(mode="multiclass", 
                                     smooth=smooth, 
                                     from_logits=True, 
                                     alpha=alpha, 
                                     beta=beta,
                                     ignore_index=0)
        self.class_loss = FocalLoss(mode="multiclass", 
                                    alpha=0.25, 
                                    gamma=2.) #GSPLoss() #TopKClassificationLoss()
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
            classification_loss = self.class_loss(inputs, targets_mask) #targets_class)
        if self.lambda_con > 0:
            #con_loss = self.con_loss(embedding, targets_class.repeat_interleave(self.time_steps))
            con_loss = self.con_loss(inputs, targets_mask)
        

        total_loss = self.lambda_seg * segmentation_loss + self.lambda_class * classification_loss + self.lambda_con * con_loss
        
        if hasattr(self, 'fire_loss'):
            tax_loss = self.fire_loss.compute_tax()
            total_loss += tax_loss
        
        return total_loss
