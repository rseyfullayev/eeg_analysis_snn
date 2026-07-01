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
                 iic_inter_weight=1.0,
                 decoupled=False):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.use_scda = iic_enabled
        self.iic_intra_weight = iic_intra_weight
        self.iic_inter_weight = iic_inter_weight
        self.decoupled = decoupled
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
        
        if self.use_scda:
            pos_weights = mask * self.iic_intra_weight
            neg_weights = neg_mask * (1.0 + dice_pair)
        else:
            pos_weights = mask.clone()
            neg_weights = neg_mask * (1.0 + dice_pair)

        if self.decoupled:
            denom_weights = neg_weights
        else:
            denom_weights = pos_weights + neg_weights

        # Numerical Stability: subtract max for LogSumExp
        if self.decoupled:
            mask_for_max = torch.where(mask.bool(), torch.full_like(similarity_matrix, -1e9), similarity_matrix)
            logits_max, _ = torch.max(mask_for_max, dim=1, keepdim=True)
        else:
            logits_max, _ = torch.max(similarity_matrix * logits_mask, dim=1, keepdim=True)
            
        logits = similarity_matrix - logits_max.detach()
        
        # Compute log softmax with dice-scaled denominator
        exp_logits = torch.exp(logits) * denom_weights
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # Mean log-likelihood over positives
        valid_anchors = pos_weights.sum(1) > 0
        if not valid_anchors.any():
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            pos_weight_sum = pos_weights[valid_anchors].sum(1)
            mean_log_prob_pos = (pos_weights[valid_anchors] * log_prob[valid_anchors]).sum(1) / pos_weight_sum
            loss = -mean_log_prob_pos.mean()
        
        current_epoch = getattr(self, 'current_epoch', 0)
        step_count = getattr(self, 'step_count', 0)
        
        with torch.no_grad():
            raw_dots = torch.matmul(features, features.T) 
            pos_dots = (raw_dots * mask).sum() / (mask.sum() + 1e-6)
            neg_dots = (raw_dots * neg_mask).sum() / (neg_mask.sum() + 1e-6)

            if not hasattr(self, 'ema_pos_dots'):
                self.ema_pos_dots = pos_dots.item()
                self.ema_neg_dots = neg_dots.item()
            else:
                alpha = 0.05
                self.ema_pos_dots = (1 - alpha) * self.ema_pos_dots + alpha * pos_dots.item()
                self.ema_neg_dots = (1 - alpha) * self.ema_neg_dots + alpha * neg_dots.item()

        if step_count > 0 and step_count % 500 == 0:
            with torch.no_grad():
                mean_pos_weight = pos_weights[mask.bool()].mean()
                mean_neg_weight = neg_weights[neg_mask.bool()].mean()
                denom_val = exp_logits.sum(1).mean().item()

                print(f"\n--- Loss Diagnostics EMA (Epoch {current_epoch} | Batch {step_count}) ---")
                print(f"EMA Pos Dot Product: {self.ema_pos_dots:.4f}")
                print(f"EMA Neg Dot Product: {self.ema_neg_dots:.4f}")
                print(f"Mean Pos Weight: {mean_pos_weight.item():.4f}")
                print(f"Mean Neg Weight: {mean_neg_weight.item():.4f}")
                print(f"Max Logit (shifted): {logits_max.mean().item():.4f}")
                print(f"Denominator (Exp Neg Sum): {denom_val:.4f}")
                print(f"Final Loss: {loss.item():.4f}\n")
                
        self.step_count = step_count + 1
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
                 temporal_decay_factor=0.999,
                 exclude_same_trial=True,
                 cosface_margin=0.0):
        super().__init__(
            masks=masks,
            temperature=temperature,
            iic_enabled=iic_enabled,
            iic_intra_weight=iic_intra_weight,
            iic_inter_weight=iic_inter_weight,
            decoupled=True,
        )
        self.temporal_decay_enabled = temporal_decay_enabled
        self.temporal_decay_factor = temporal_decay_factor
        self.exclude_same_trial = exclude_same_trial
        self.cosface_margin = cosface_margin

    def forward(self,
                query_features,
                key_features,
                labels,
                subject_labels,
                queue_features=None,
                queue_labels=None,
                queue_subject_labels=None,
                queue_ages=None,
                query_video_ids=None,
                query_timestamps=None,
                queue_video_ids=None,
                queue_timestamps=None):
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
        # Track video_ids and timestamps for candidates (keys from current batch)
        candidate_video_ids = [query_video_ids if query_video_ids is not None else torch.full_like(labels, -1)]
        candidate_timestamps = [query_timestamps if query_timestamps is not None else torch.full_like(labels, -1)]

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

                # Temporal identity for queue entries
                if queue_video_ids is not None:
                    candidate_video_ids.append(queue_video_ids[valid].to(device))
                else:
                    candidate_video_ids.append(torch.full_like(q_lbl, -1))
                if queue_timestamps is not None:
                    candidate_timestamps.append(queue_timestamps[valid].to(device))
                else:
                    candidate_timestamps.append(torch.full_like(q_lbl, -1))

        all_features = torch.cat(candidate_features, dim=0)  # (M, D)
        all_labels = torch.cat(candidate_labels, dim=0)      # (M,)
        all_subject_labels = torch.cat(candidate_subject_labels, dim=0)  # (M,)
        all_ages = torch.cat(candidate_ages, dim=0)          # (M,)
        all_video_ids = torch.cat(candidate_video_ids, dim=0)  # (M,)
        all_timestamps = torch.cat(candidate_timestamps, dim=0)  # (M,)
        

        # Compute raw cosine similarities (before temperature)
        cos_sim = torch.matmul(q, all_features.T)  # (B, M) — values in [-1, 1]

        pos_mask = (labels.unsqueeze(1) == all_labels.unsqueeze(0)).float()
        neg_mask = 1.0 - pos_mask
        same_subj_mask = (subject_labels.unsqueeze(1) == all_subject_labels.unsqueeze(0)).float()
        diff_subj_mask = 1.0 - same_subj_mask

        # --- CosFace Angular Margin ---
        # Subtract margin from positive cosine similarities IN COSINE SPACE (before temperature)
        # This is the correct formulation: cos(θ) - m for positives, cos(θ) for negatives
        if self.cosface_margin > 0:
            cos_sim_margin = cos_sim - self.cosface_margin * pos_mask
        else:
            cos_sim_margin = cos_sim

        # Now apply temperature scaling to get logits
        logits = cos_sim_margin / self.temperature  # (B, M)
        # Unpenalized logits for the denominator (negatives don't get margin)
        logits_neg = cos_sim / self.temperature

        # Mask out ALL positives (same class) from the max-shift so they don't skew the partition
        mask_for_max = torch.where(pos_mask.bool(), torch.full_like(logits_neg, -1e9), logits_neg)
        logits_max, _ = torch.max(mask_for_max, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        logits_neg = logits_neg - logits_max.detach()

        if self.use_scda:
            # 1. Domain Adaptation Positives: Labels equal, Subjects differ
            pos_weights = pos_mask * diff_subj_mask

            # 2. Hard Negatives: Labels differ, Subjects equal -> Weight = (1.0 + Dice)
            # 3. Normal Negatives: Labels differ, Subjects differ -> Weight = 1.0
            dice_pair = self.sim_score[labels][:, all_labels]
            
            hard_neg_mask = neg_mask * same_subj_mask
            normal_neg_mask = neg_mask * diff_subj_mask
            
            neg_weights = (hard_neg_mask * (1.0 + dice_pair)) + normal_neg_mask
        else:
            # Standard SupCon with Dice-weighted negatives
            dice_pair = self.sim_score[labels][:, all_labels]
            pos_weights = pos_mask.clone()
            neg_weights = neg_mask * (1.0 + dice_pair)

        if self.temporal_decay_enabled:
            # Apply temporal decay: w_temporal = temporal_decay_factor ^ age
            temporal_weights = self.temporal_decay_factor ** all_ages  # (M,)
            neg_weights = neg_weights * temporal_weights.unsqueeze(0)
            pos_weights = pos_weights * temporal_weights.unsqueeze(0)

        # --- Trial Exclusion Mask (Leave-One-Trial-Out) ---
        # Zero out POSITIVES from the same trial.  Under weak supervision
        # (1 trial = 1 label) the neg_mask already excludes same-label pairs,
        # so masking negatives here was a no-op.  Instead, we prevent same-trial
        # windows from being pulled together as positives.  This combats the
        # "gaslighting" problem: not every window in a trial is equally
        # emotional, and anchoring to intra-trial climax peaks drags
        # neutral-ish windows toward a trial-specific representation.
        # Same-trial windows become ghosts (neither pos nor neg), and the
        # model is forced to learn cross-trial emotion agreement only.
        if self.exclude_same_trial and query_video_ids is not None:
            q_vid = query_video_ids.view(-1, 1)    # (B, 1)
            c_vid = all_video_ids.view(1, -1)       # (1, M)

            same_video = (q_vid == c_vid)  # (B, M)
            pos_weights = pos_weights * (~same_video).float()

        # Decoupled denominator: negatives only (no positive terms in partition).
        # Use unpenalized logits for denominator so margin only affects numerator
        neg_partition = torch.clamp((torch.exp(logits_neg) * neg_weights).sum(dim=1, keepdim=True), min=1e-8)
        pos_log_prob = logits - torch.log(neg_partition)

        valid_anchors = pos_weights.sum(dim=1) > 0
        if not valid_anchors.any():
            loss_val = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            pos_weight_sum = pos_weights[valid_anchors].sum(dim=1)
            mean_log_prob_pos = (pos_weights[valid_anchors] * pos_log_prob[valid_anchors]).sum(dim=1) / pos_weight_sum
            loss_val = -mean_log_prob_pos.mean()
        
        current_epoch = getattr(self, 'current_epoch', 0)
        step_count = getattr(self, 'step_count', 0)
        
        with torch.no_grad():
            # cos_sim already contains raw cosine similarities (no temperature)
            pos_dots = (cos_sim * pos_mask).sum() / (pos_mask.sum() + 1e-6)
            neg_dots = (cos_sim * neg_mask).sum() / (neg_mask.sum() + 1e-6)

            if not hasattr(self, 'ema_pos_dots'):
                self.ema_pos_dots = pos_dots.item()
                self.ema_neg_dots = neg_dots.item()
            else:
                alpha = 0.05
                self.ema_pos_dots = (1 - alpha) * self.ema_pos_dots + alpha * pos_dots.item()
                self.ema_neg_dots = (1 - alpha) * self.ema_neg_dots + alpha * neg_dots.item()

        if step_count > 0 and step_count % 500 == 0:
            with torch.no_grad():
                mean_pos_weight = pos_weights[pos_mask.bool()].mean()
                mean_neg_weight = neg_weights[neg_mask.bool()].mean()

                print(f"\n--- SupMoCo Diagnostics EMA (Epoch {current_epoch} | Batch {step_count}) ---")
                print(f"EMA Pos Dot Product: {self.ema_pos_dots:.4f}")
                print(f"EMA Neg Dot Product: {self.ema_neg_dots:.4f}")
                print(f"Mean Pos Weight: {mean_pos_weight.item():.4f}")
                print(f"Mean Neg Weight: {mean_neg_weight.item():.4f}")
                print(f"Max Logit (shifted): {logits_max.mean().item():.4f}")
                print(f"Denominator (Exp Neg Sum): {neg_partition.mean().item():.4f}")
                if self.exclude_same_trial and query_video_ids is not None:
                    ghosted_count = (same_video & pos_mask.bool()).sum().item()
                    total_pos_pairs = pos_mask.sum().item()
                    print(f"LOTO Ghosted: {ghosted_count:.0f} / {total_pos_pairs:.0f} pos pairs excluded (cross-trial only)")
                print(f"Final Loss: {loss_val.item():.4f}\n")
                
        self.step_count = step_count + 1
        return loss_val


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

class MultiKernelMMDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def compute_median_distance(self, features):
        """Compute median pairwise distance across all features for dynamic bandwidth."""
        with torch.no_grad():
            distances = torch.cdist(features, features, p=2.0)
            upper_tri_indices = torch.triu_indices(distances.size(0), distances.size(1), offset=1)
            pairwise_dists = distances[upper_tri_indices[0], upper_tri_indices[1]]
            if pairwise_dists.numel() == 0:
                return 1.0 # fallback
            median_dist = torch.median(pairwise_dists).item()
            return median_dist if median_dist > 1e-5 else 1.0

    def forward(self, features, domain_labels, class_labels=None):
        """Calculates Multi-Kernel MMD across all pairs of domains natively present in the batch."""

        features = F.normalize(features, dim=1) # L2 Normalization ensuring that Dual Loss lives in same hypersphere

        # Safety: bail out if features are degenerate (NaN/Inf from collapse or overflow)
        if not torch.isfinite(features).all():
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        unique_domains = torch.unique(domain_labels)
        if len(unique_domains) < 2:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        sigma_median = self.compute_median_distance(features)
        bandwidths = [sigma_median / 4.0, sigma_median / 2.0, sigma_median, sigma_median * 2.0, sigma_median * 4.0]

        # Precompute kernel matrix across ALL features once for efficiency
        dist_sq = torch.cdist(features, features, p=2.0).pow(2)
        kernel_matrix = torch.zeros_like(dist_sq)
        for sigma in bandwidths:
            gamma = 1.0 / (2 * (sigma ** 2))
            kernel_matrix += torch.exp(-gamma * dist_sq)
            
        B = features.size(0)
        max_class = class_labels.max()+1
        group_ids = domain_labels * max_class + class_labels
        
        unique_groups, inverse_indices = torch.unique(group_ids, return_inverse=True)
        G = unique_groups.size(0)

        if G<2:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        H = F.one_hot(inverse_indices, num_classes=G).to(features.dtype)

        group_counts = H.sum(dim=0)
        H_norm = H / torch.clamp(group_counts, min=1.0).unsqueeze(0)

        K_groups = torch.mm(torch.mm(H_norm.t(), kernel_matrix), H_norm)

        group_domains = unique_groups // max_class
        group_classes = unique_groups % max_class

        same_class = (group_classes.unsqueeze(1) == group_classes.unsqueeze(0)).float()
        diff_domain = (group_domains.unsqueeze(1) != group_domains.unsqueeze(0)).float()

        valid_pairs = torch.triu(same_class * diff_domain, diagonal=1)
        
        K_self = torch.diag(K_groups)
        MMD_matrix = K_self.unsqueeze(1) + K_self.unsqueeze(0) - 2.0*K_groups
        num_pairs = valid_pairs.sum()

        if num_pairs>0:
            mmd_loss = (MMD_matrix * valid_pairs).sum() / num_pairs
        else:
            mmd_loss = torch.tensor(0.0, device=features.device, requires_grad=True)
        
        return mmd_loss

class SubjectEraserLoss(nn.Module):
    def __init__(self, beta=1e-3, gamma=0.1, dann_weight=1.0):
        super(SubjectEraserLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.beta = beta
        self.max_beta = beta
        self.gamma = gamma
        self.dann_weight = dann_weight

    def forward(self, logits, targets_class, mu, logvar, h_emo, h_dmn, dann_logits=None, subj_logits=None, targets_domain=None):
        # 1. Classification Loss
        cls_loss = self.ce(logits, targets_class)
        
        # 2. VIB KL Divergence Loss
        # KL(N(mu, sigma^2) || N(0, 1)) = -0.5 * sum(1 + logvar - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        
        # 3. Orthogonality Loss
        # Soft Frobenius orthogonalization penalty (squared cosine similarity)
        cos_sim = F.cosine_similarity(h_emo, h_dmn, dim=1)
        ortho_loss = torch.mean(cos_sim.pow(2))
        
        # 4. DANN (Subject Erasure) and Explicit Subject Routing Loss
        dann_loss = 0.0
        subj_loss = 0.0
        if targets_domain is not None:
            valid_mask = targets_domain != -1
            if valid_mask.any():
                if dann_logits is not None:
                    dann_loss = self.ce(dann_logits[valid_mask], targets_domain[valid_mask])
                if subj_logits is not None:
                    subj_loss = self.ce(subj_logits[valid_mask], targets_domain[valid_mask])
                
        total_loss = cls_loss + self.beta * kl_loss + self.gamma * ortho_loss + self.dann_weight * dann_loss + subj_loss
        
        return total_loss, {
            'loss_cls': cls_loss.item(),
            'loss_kl': kl_loss.item(),
            'loss_ortho': ortho_loss.item(),
            'loss_dann': dann_loss.item() if isinstance(dann_loss, torch.Tensor) else dann_loss,
            'loss_subj': subj_loss.item() if isinstance(subj_loss, torch.Tensor) else subj_loss
        }
