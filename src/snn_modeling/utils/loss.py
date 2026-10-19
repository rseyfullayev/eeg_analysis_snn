import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torch.nn.functional as F



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
        
        # Mean log-likelihood over positives. A fully ghosted batch is a masking/data
        # bug, not a training step — raise instead of emitting a silent zero loss
        # (audit §3.5 row 8; interiority guarantees this cannot fire on healthy data).
        valid_anchors = pos_weights.sum(1) > 0
        if not valid_anchors.any():
            raise RuntimeError(
                f"ContrastiveLoss: no anchor retains a positive (raw pos pairs={mask.sum().item()}, "
                f"batch={batch_size}). Check LOTO masking / labels; refusing zero-loss step.")
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
            raise RuntimeError(
                f"SupMoCoLoss: every anchor lost all cross-trial positives "
                f"(pos pairs={pos_mask.sum().item()}, batch={q.shape[0]}, "
                f"candidates={all_features.shape[0]}). Check LOTO masking / queue video ids; "
                "refusing zero-loss step.")
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
