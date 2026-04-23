# CITE https://github.com/HobbitLong/SupContrast/tree/master

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.snn_modeling.utils.loss import FullHybridLoss, TopKClassificationLoss, ContrastiveLoss, SupMoCoLoss, MultiKernelMMDLoss
from src.snn_modeling.utils.optim import Muon, HybridOptimizer, HybridScheduler
from src.snn_modeling.dataloader.dataset import SWEEPDataset, PKSampler
from src.snn_modeling.utils.supmoco import SupMoCoState, build_momentum_encoder, momentum_update
import os
import gc
from datetime import datetime
from sklearn.metrics import balanced_accuracy_score
import snntorch as snn
import segmentation_models_pytorch as smp
import wandb
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import numpy as np
from tqdm import tqdm
from src.snn_modeling.utils.utils import initialize_network
from src.snn_modeling.utils.augmentations import VideoTemporalMasking, GaussianNoise, FrequencyDropout, SignalJitter, VideoRandomErasing, SpatialDropout
from src.snn_modeling.layers.neurons import ALIF
from src.snn_modeling.models.unet import SpikingMobileNetProjector
from omegaconf import OmegaConf
# Ignore the specific sklearn warning about missing classes
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
warnings.filterwarnings("ignore", message="A single label was found in 'y_true' and 'y_pred'. For the confusion matrix to have the correct shape, use the 'labels' parameter to pass all known labels.")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

#torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True
def save_checkpoint(model, optimizer, scheduler, epoch, acc, dice, path="best_sweepnet.pt",
                    momentum_model=None, supmoco_state=None):
    payload = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy': acc,
        'dice': dice,
        'momentum_model_state_dict': momentum_model.state_dict() if momentum_model is not None else None,
        'supmoco_state_dict': supmoco_state.state_dict() if supmoco_state is not None else None,
    }
    torch.save(payload, path)
    print(f"Saved checkpoint at Epoch {epoch} (Acc: {acc:.4f}, Dice: {dice:.4f})")


def validate(model, val_loader, criterion, device, threshold=0.5, only_classification=False):
    if only_classification:
        return 0, 0, 0, 0, 0, 0, 0
   
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []

    tp_tot, fp_tot, fn_tot, tn_tot = 0, 0, 0, 0

    val_loop = tqdm(val_loader, desc=f"Validation", unit="batch")
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loop):
            inputs, targets, labels = batch[0], batch[1], batch[2]
            inputs, targets, labels = inputs.to(device),  targets.to(device), labels.to(device)
            
            K_bag = None
            if inputs.dim() == 6:
                B, K_bag, T, C, H, W = inputs.shape
                inputs = inputs.view(B * K_bag, T, C, H, W)
            else:
                B, T, C, H, W = inputs.shape
                
            inputs = inputs.permute(1, 0, 2, 3, 4)
            
            outputs = model(inputs, K=K_bag)

            if K_bag is not None and outputs.shape[0] == B * K_bag:
                outputs = outputs.view(B, K_bag, *outputs.shape[1:]).mean(dim=1)

            loss = criterion(outputs, targets, labels)
            B, C, H, W = outputs.shape
            
            probs = torch.softmax(outputs, dim=1)
            energy_logits = probs[:, 1:, :, :].sum(dim=(2, 3))
            preds_map = torch.argmax(probs, dim=1)
            tp, fp, fn, tn = smp.metrics.get_stats(
                preds_map,
                targets, 
                mode='multiclass', 
                num_classes=6
            )
            tp_tot += tp[:, 1:].sum().item()
            fp_tot += fp[:, 1:].sum().item()
            fn_tot += fn[:, 1:].sum().item()
            tn_tot += tn[:, 1:].sum().item()
            preds = energy_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())


            val_loss += loss.item()
            val_loop.set_postfix(loss=loss.item())
            
            
            
            # Cleanup tensors to free memory
            del inputs, labels, outputs, loss, preds
            del preds_map, targets, tp, fp, fn, tn
                
    # Clear CUDA cache after validation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
                
            
    avg_loss = val_loss / len(val_loader)
    accuracy = correct / total
    balanced_acc = balanced_accuracy_score(all_targets, all_preds)
    
    eps = 1e-7
    dice_score = (2 * tp_tot) / (2 * tp_tot + fp_tot + fn_tot + eps)
    iou_score = tp_tot / (tp_tot + fp_tot + fn_tot + eps)
    precision = tp_tot / (tp_tot + fp_tot + eps)
    recall = tp_tot / (tp_tot + fn_tot + eps)

    return avg_loss, accuracy, balanced_acc, dice_score, iou_score, precision, recall

def log_visuals(model, val_loader, device, writer, epoch, threshold=0.5):

    model.eval()
    
    try:
        data_iter = iter(val_loader)
        batch = next(data_iter)
        inputs, targets, labels = batch[0], batch[1], batch[2]
    except StopIteration:
        return 

    inputs, targets, labels = inputs.to(device), targets.to(device), labels.to(device)
    inputs_snn = inputs.permute(2, 0, 1, 3, 4)

    with torch.no_grad():
        outputs = model(inputs_snn)
        probs = torch.sigmoid(outputs)

    num_samples = min(4, inputs.shape[0])
    
    for idx in range(num_samples):
        true_class = labels[idx].item()
    
        img_input = inputs[idx].mean(dim=(0, 1)).cpu().numpy()
        min_v, max_v = img_input.min(), img_input.max()
        if max_v - min_v > 1e-7:
            img_input = (img_input - min_v) / (max_v - min_v)
        else:
            img_input = np.zeros_like(img_input)

        img_target = targets[idx, true_class].cpu().numpy()
        img_pred = probs[idx, true_class].cpu().numpy()
        img_pred_bin = (img_pred > threshold).astype(float)
        caption_text = f"Class {true_class} | Sample {idx} | Ep {epoch}"
        
        writer.add_image(f"Vis/Target_{idx}", img_target[None, ...], epoch)
        writer.add_image(f"Vis/Pred_{idx}", img_pred_bin[None, ...], epoch)
        
        wandb.log({
            f"Visuals/Sample_{idx}": [
                wandb.Image(img_target, caption=f"Target {caption_text})"),
                wandb.Image(img_pred_bin, caption=f"Hard Pred (Thresh {threshold})")
            ]
        }, step=epoch)
    
    # Cleanup tensors
    del inputs, targets, labels, outputs, probs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def freeze_bn_stats(module):
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
            module.eval()

def create_optimizer(model, loss_fn, config, low_encoder_lr=False):
    lr = config.training.get('learning_rate', 1e-3)
    use_muon = config.training.get('use_muon', False)
    weight_decay = config.training.get('weight_decay', 1e-4)

    # ── Identify special (no-decay) parameter IDs ──
    no_decay_id = set()
    classess_names = (snn.Leaky, snn.Synaptic, snn.Alpha, 
                      ALIF, TopKClassificationLoss, 
                      nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, 
                      nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)
    # ── Identify depthwise conv weight IDs (ban from Muon) ──
    # Depthwise convs have groups == in_channels; their per-channel kernels are
    # independent, so Newton-Schulz orthogonalization across them is invalid.
    depthwise_id = set()
    for m in model.modules():
        if isinstance(m, classess_names):
            for param in m.parameters(recurse=False):
                no_decay_id.add(id(param))    
        if hasattr(m, 'bias') and m.bias is not None:
            no_decay_id.add(id(m.bias))
        if hasattr(m, 'gain') and m.gain is not None:
             no_decay_id.add(id(m.gain))
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if m.groups == m.in_channels and m.groups > 1:
                depthwise_id.add(id(m.weight))
    for m in loss_fn.modules():
        if isinstance(m, classess_names):
            for param in m.parameters(recurse=False):
                no_decay_id.add(id(param))      

    # ── Classify every parameter ──
    _SPECIAL_KEYS = ('alpha', 'beta', 'slope', 'decay', 'gamma', 'recurrent')

    # AdamW buckets (always used)
    adam_base_params = []
    adam_encoder_params = []
    adam_no_decay_params = []
    time_params = []
    threshold_params = []

    # Muon buckets (only populated when use_muon=True)
    muon_params = []
    muon_encoder_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_special_key = any(k in name for k in _SPECIAL_KEYS)
        is_no_decay = id(param) in no_decay_id
        is_threshold = 'threshold' in name

        # 1. Time-constant params → AdamW with reduced LR
        if is_special_key:
            time_params.append(param)
            print(f"[AdamW | Time] {name}")
        # 2. Threshold params → AdamW
        elif is_threshold:
            threshold_params.append(param)
            print(f"[AdamW | Threshold] {name}")
        # 3. No-decay params (norms, biases, gains) → AdamW
        elif is_no_decay:
            adam_no_decay_params.append(param)
            print(f"[AdamW | No-Decay] {name}")
        # 4. Muon-eligible: ndim >= 2, not special, not depthwise conv
        elif use_muon and param.ndim >= 2 and id(param) not in depthwise_id:
            if low_encoder_lr and 'encoder' in name:
                muon_encoder_params.append(param)
                print(f"[Muon | Encoder] {name}")
            else:
                muon_params.append(param)
                print(f"[Muon] {name}")
        # 5. Everything else (including depthwise conv weights) → AdamW base
        else:
            if low_encoder_lr and 'encoder' in name:
                adam_encoder_params.append(param)
                print(f"[AdamW | Encoder Low-LR] {name}")
            else:
                adam_base_params.append(param)
                print(f"[AdamW | Base] {name}")

    # Loss function parameters → always AdamW
    for name, param in loss_fn.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in no_decay_id:
            adam_no_decay_params.append(param)
            print(f"[AdamW | Loss No-Decay] {name}")
        else:
            adam_base_params.append(param)
            print(f"[AdamW | Loss Base] {name}")

    # ── Build AdamW ──
    adam_groups = [
        {'params': adam_base_params, 'lr': lr, 'weight_decay': weight_decay},
        {'params': adam_encoder_params, 'lr': lr * 1e-2, 'weight_decay': weight_decay},
        {'params': adam_no_decay_params, 'lr': lr, 'weight_decay': 0.0},
        {'params': time_params, 'lr': lr * 0.5, 'weight_decay': 0.0},
        {'params': threshold_params, 'lr': lr * 1.0, 'weight_decay': 0.0},
    ]
    # Filter out empty groups
    adam_groups = [g for g in adam_groups if len(g['params']) > 0]
    optimizer_adamw = optim.AdamW(adam_groups, betas=(0.9, 0.999)) if adam_groups else None

    # ── Build Muon (if requested) ──
    optimizer_muon = None
    if use_muon:
        muon_lr = config.training.get('muon_lr', 0.02)
        muon_momentum = config.training.get('muon_momentum', 0.95)
        muon_groups = [
            {'params': muon_params, 'lr': muon_lr, 'weight_decay': weight_decay, 'momentum': muon_momentum},
            {'params': muon_encoder_params, 'lr': muon_lr * 1e-2, 'weight_decay': weight_decay, 'momentum': muon_momentum},
        ]
        muon_groups = [g for g in muon_groups if len(g['params']) > 0]
        if muon_groups:
            optimizer_muon = Muon(muon_groups, lr=muon_lr, weight_decay=weight_decay, momentum=muon_momentum)
            n_muon = sum(p.numel() for g in muon_groups for p in g['params'])
            n_adam = sum(p.numel() for g in adam_groups for p in g['params']) if adam_groups else 0
            print(f"\n=== Muon+AdamW Hybrid ===")
            print(f"  Muon params:  {n_muon:,}")
            print(f"  AdamW params: {n_adam:,}")
            print(f"  Muon LR: {muon_lr} | AdamW LR: {lr}\n")

    if use_muon and optimizer_muon is not None:
        return HybridOptimizer(optimizer_muon, optimizer_adamw)
    else:
        return optimizer_adamw


def create_hybrid_scheduler(optimizer, T_max, eta_min=1e-6):
    """Create a CosineAnnealingLR scheduler, handling both HybridOptimizer and plain optimizer."""
    if isinstance(optimizer, HybridOptimizer):
        sched_muon = None
        sched_adamw = None
        if optimizer.opt_muon is not None:
            sched_muon = CosineAnnealingLR(optimizer.opt_muon, T_max=T_max, eta_min=eta_min)
        if optimizer.opt_adamw is not None:
            sched_adamw = CosineAnnealingLR(optimizer.opt_adamw, T_max=T_max, eta_min=eta_min)
        return HybridScheduler(sched_muon, sched_adamw)
    else:
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)



@torch.no_grad()
def compute_queue_knn_accuracy(model, val_loader, device, supmoco_state, k=5):
    """Probe representation quality by kNN retrieval against the SupMoCo memory queue.
    
    For each val sample, find its k nearest neighbors in the queue via cosine
    similarity, then majority-vote the label. Returns accuracy.
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    for batch in val_loader:
        inputs, labels = batch[0], batch[2]
        inputs, labels = inputs.to(device), labels.to(device)

        K_bag = None
        if inputs.dim() == 6:
            B, K_bag, T, C, H, W = inputs.shape
            inputs = inputs.view(B * K_bag, T, C, H, W)
        else:
            B = inputs.shape[0]

        inputs = inputs.permute(1, 0, 2, 3, 4)
        features = model(inputs, K=K_bag)
        if K_bag is not None and features.shape[0] == B * K_bag:
            features = features.view(B, K_bag, *features.shape[1:]).mean(dim=1)
        features = F.normalize(features, dim=1, eps=1e-6)

        all_embeddings.append(features)
        all_labels.append(labels)

    val_emb = torch.cat(all_embeddings, dim=0)    # (N_val, D)
    val_labels = torch.cat(all_labels, dim=0)      # (N_val,)

    # Get filled queue
    queue_feats, queue_labels, _, _, _, _ = supmoco_state.get_queue()
    if queue_feats.shape[0] == 0:
        return 0.0

    queue_feats = F.normalize(queue_feats, dim=1, eps=1e-6)

    # Cosine similarity: (N_val, Q)
    sim = torch.matmul(val_emb, queue_feats.T)
    _, topk_indices = sim.topk(k, dim=1)            # (N_val, k)
    topk_labels = queue_labels[topk_indices]         # (N_val, k)

    # Majority vote
    preds = torch.mode(topk_labels, dim=1).values    # (N_val,)
    acc = (preds == val_labels).float().mean().item()

    return acc


def compute_linear_eval_accuracy(model, online_evaluator, val_loader, device):
    """Online linear evaluation accuracy on the validation set.
    
    Extracts backbone features (detached from the contrastive graph),
    pushes them through the online linear head, and returns balanced accuracy.
    """
    model.eval()
    online_evaluator.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch[0], batch[2]
            inputs, labels = inputs.to(device), labels.to(device)

            K_bag = None
            if inputs.dim() == 6:
                B, K_bag, T, C, H, W = inputs.shape
                inputs = inputs.view(B * K_bag, T, C, H, W)
            else:
                B = inputs.shape[0]

            inputs = inputs.permute(1, 0, 2, 3, 4)
            features = model.extract_features(inputs, K=K_bag)
            if K_bag is not None and features.shape[0] == B * K_bag:
                features = features.view(B, K_bag, *features.shape[1:]).mean(dim=1)

            logits = online_evaluator(features)
            preds = logits.argmax(dim=1)
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Balanced accuracy: mean per-class recall
    num_classes = logits.shape[1]
    per_class_acc = []
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            per_class_acc.append((all_preds[mask] == c).float().mean().item())
    bal_acc = sum(per_class_acc) / max(len(per_class_acc), 1)
    return bal_acc


def training_loop(phase, 
                  start_epoch, 
                  epochs, 
                  best_acc, 
                  best_dice, 
                  model, 
                  device, 
                  train_loader, 
                  val_loader, 
                  loss_fn, 
                  optimizer, 
                  scheduler, 
                  writer, 
                  checkpoint_dir,
                  freeze_bn=False,
                  use_supmoco=False,
                  moco_momentum=0.999,
                  momentum_model=None,
                  supmoco_state=None,
                  unfreeze_epoch=-1,
                  accumulation_steps=1,
                  dann_weight=1.0,
                  dann_alpha=1.0,
                  subj_remapper=None,
                  probe_interval=5,
                  use_mmd=False,
                  lambda_mmd=0.0):
    
    # Phase 1A trackers
    best_train_loss = float('inf') if phase == '1a' else None
    best_linear_acc = 0.0

    #temp_mix = TemporalMix()

    for epoch in range(start_epoch, epochs):
        model.train()
        loss_fn.current_epoch = epoch
        loss_fn.step_count = 0

        if unfreeze_epoch > 0:
            if epoch < unfreeze_epoch:
                for param in model.encoder.parameters():
                    param.requires_grad = False
            elif epoch == unfreeze_epoch:
                print("--- Unfreezing Encoder ---")
                for param in model.encoder.parameters():
                    param.requires_grad = True

        if freeze_bn:
            model.encoder.apply(freeze_bn_stats)
        train_loss = 0.0
        train_linear_correct = 0
        train_linear_total = 0
        train_dann_correct = 0
        train_dann_total = 0
        dann_loss_total = 0.0
        train_loop = tqdm(train_loader, desc=f"Phase {phase} Epoch {epoch+1}/{epochs}", unit="batch")
        for batch_idx, batch in enumerate(train_loop):
            # Handle both augmented (8 items) and non-augmented (6 items) returns
            # Augmented: (inp1, inp2, targets, targets_c, subject_labels, bag_id, video_id, timestamp)
            # Non-augmented: (inputs, targets, targets_c, bag_id, video_id, timestamp)
            if len(batch) == 8:
                inp1, inp2, targets, targets_c, subject_labels, _, video_ids, timestamps = batch
                video_ids = video_ids.to(device)
                timestamps = timestamps.to(device)
                if phase in [1, '1a', '1b']:
                    inp1, inp2 = inp1.to(device), inp2.to(device)
                    subject_labels = subject_labels.to(device)
                    if use_supmoco:
                        q_inputs = inp1
                        k_inputs = inp2
                    else:
                        # Contrastive baseline: concatenate both views
                        inputs = torch.cat([inp1, inp2], dim=0)
            elif len(batch) == 7:
                inp1, inp2, targets, targets_c, _, video_ids, timestamps = batch
                video_ids = video_ids.to(device)
                timestamps = timestamps.to(device)
                if phase in [1, '1a', '1b']:
                    if use_supmoco:
                        raise ValueError("SupMoCo requires augmented batches that include subject labels (len(batch) == 8).")
                    inp1, inp2 = inp1.to(device), inp2.to(device)
                    if use_supmoco:
                        q_inputs = inp1
                        k_inputs = inp2
                    else:
                        inputs = torch.cat([inp1, inp2], dim=0)
                else:
                    # Non-contrastive phases: just use first view
                    inputs = inp1.to(device)
            else:
                inputs, targets, targets_c, _, video_ids, timestamps = batch
                video_ids = video_ids.to(device)
                timestamps = timestamps.to(device)
                inputs = inputs.to(device)
                if phase in [1, '1a', '1b'] and use_supmoco:
                    raise ValueError("SupMoCo requires dual-view augmented batches (len(batch) >= 7).")
                
            targets, targets_c = targets.to(device), targets_c.to(device)
            
            if phase in [1, '1a', '1b'] and use_supmoco:
                def _prepare_view(view):
                    K_local = None
                    if view.dim() == 6:
                        B_local, K_local, T_local, C_local, H_local, W_local = view.shape
                        view = view.view(B_local * K_local, T_local, C_local, H_local, W_local)
                    else:
                        B_local, T_local, C_local, H_local, W_local = view.shape
                    view = view.permute(1, 0, 2, 3, 4)
                    return view, B_local, K_local

                q_inputs, B, K_bag = _prepare_view(q_inputs)
                k_inputs, _, _ = _prepare_view(k_inputs)

                # Extract backbone features, then project for contrastive loss
                backbone_feats = model.extract_features(q_inputs, K=K_bag)
                if K_bag is not None and backbone_feats.shape[0] == B * K_bag:
                    backbone_feats = backbone_feats.view(B, K_bag, *backbone_feats.shape[1:]).mean(dim=1)
                outputs = model.classifier(backbone_feats)

                with torch.no_grad():
                    key_features = momentum_model(k_inputs, K=K_bag)
                    if K_bag is not None and key_features.shape[0] == B * K_bag:
                        key_features = key_features.view(B, K_bag, *key_features.shape[1:]).mean(dim=1)
                    key_features = F.normalize(key_features, dim=1, eps=1e-6)

                queue_features, queue_labels, queue_subject_labels, queue_ages, q_video_ids, q_timestamps = supmoco_state.get_queue()
                loss = loss_fn(outputs, key_features, targets_c, subject_labels,
                               queue_features, queue_labels, queue_subject_labels, queue_ages,
                               query_video_ids=video_ids, query_timestamps=timestamps,
                               queue_video_ids=q_video_ids, queue_timestamps=q_timestamps)

                # --- Offline Probe happens below in validation ---
                target_lr_momentum = 0.75
                # --- DANN (Domain Adversarial Neural Network) ---
                if getattr(model, 'use_dann', False):
                    print("###########################################")
                    print("DANN is enabled")
                    print("###########################################")
                    # Ganin et al. (2015) Alpha Annealing Schedule
                    # Progress p smoothly moves from 0 to 1 over the course of training phases

                    plateau_ratio = math.acos(2.0 * target_lr_momentum - 1.0) / math.pi
                    plateau_epoch = int(epochs * plateau_ratio)

                    current_step = epoch * len(train_loader) + batch_idx
                    plateau_steps = plateau_epoch * len(train_loader)

                    p = min(1.0, current_step / plateau_steps)
                    annealed_alpha = (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0) * dann_alpha

                    # Update GRL alpha if dynamically given
                    if hasattr(model, 'dann_head') and hasattr(model.dann_head[0], 'alpha'):
                        model.dann_head[0].alpha = annealed_alpha

                    subj_preds = model.dann_head(backbone_feats)
                    
                    if subj_remapper is not None:
                        mapped_labels = [subj_remapper.get(s.item() if hasattr(s, 'item') else s, 0) for s in subject_labels]
                        mapped_tensor = torch.tensor(mapped_labels, device=device, dtype=torch.long)
                    else:
                        mapped_tensor = subject_labels

                    loss_dann = F.cross_entropy(subj_preds, mapped_tensor)
                    loss = loss + (dann_weight * loss_dann)
                    
                    with torch.no_grad():
                        train_dann_correct += (subj_preds.argmax(dim=1) == mapped_tensor).sum().item()
                        train_dann_total += mapped_tensor.size(0)
                        dann_loss_total += (loss_dann.item() * targets_c.size(0))

                # --- MMD Domain Expansion ---
                if use_mmd:
                    if not hasattr(model, 'mmd_fn'):
                        print("###########################################")
                        print("MMD is enabled")
                        print("###########################################")
                        model.mmd_fn = MultiKernelMMDLoss()
                    
                    if subj_remapper is not None:
                        mapped_labels = [subj_remapper.get(s.item() if hasattr(s, 'item') else s, 0) for s in subject_labels]
                        mapped_tensor = torch.tensor(mapped_labels, device=device, dtype=torch.long)
                    else:
                        mapped_tensor = subject_labels
                        
                    loss_mmd = model.mmd_fn(backbone_feats, mapped_tensor.squeeze().long(), targets_c.squeeze().long())

                    # Ganin-like scheduler for MMD lambda
                    plateau_ratio = math.acos(2.0 * target_lr_momentum - 1.0) / math.pi
                    plateau_epoch = int(epochs * plateau_ratio)

                    current_step = epoch * len(train_loader) + batch_idx
                    plateau_steps = plateau_epoch * len(train_loader)

                    p = min(1.0, current_step / plateau_steps)

                    annealed_lambda_mmd = (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0) * lambda_mmd
                    
                    loss = loss + (annealed_lambda_mmd * loss_mmd)
            else:
                # Check if using bag-level 6D inputs: [B, K, T, C, H, W]
                K_bag = None
                if inputs.dim() == 6:
                    B, K_bag, T, C, H, W = inputs.shape
                    # Flatten Bags into Batch dimension for the SNN Encoder
                    inputs = inputs.view(B * K_bag, T, C, H, W)
                else:
                    B, T, C, H, W = inputs.shape
                
                # SNN requires Time to be dimension 0: [T, Batch, C, H, W]
                inputs = inputs.permute(1,0,2,3,4) 
                outputs = model(inputs, K=K_bag)

                # If the model didn't internally reduce K (e.g. Phase 2 UNet)
                if K_bag is not None and outputs.shape[0] == B * K_bag:
                    # Average all windows in the bag so it matches the B targets
                    outputs = outputs.view(B, K_bag, *outputs.shape[1:]).mean(dim=1)

                if phase in [1, '1a', '1b']:
                    f1, f2 = torch.split(outputs, [B//2, B//2], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    loss = loss_fn(features, targets_c)
                else:
                    loss = loss_fn(outputs, targets, targets_c)
           
            loss = loss / accumulation_steps
            loss.backward()
            
            is_step = ((batch_idx + 1) % accumulation_steps == 0) or ((batch_idx + 1) == len(train_loader))

            if is_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if phase in [1, '1a', '1b'] and use_supmoco:
                    momentum_update(model, momentum_model, moco_momentum)
                optimizer.zero_grad(set_to_none=True)

            if phase in [1, '1a', '1b'] and use_supmoco:
                with torch.no_grad():
                    supmoco_state.enqueue(key_features, targets_c, subject_labels,
                                          video_ids=video_ids, timestamps=timestamps)
            
            if phase not in [1, '1a', '1b']:
                with torch.no_grad():
                    for name, param in model.decoder.named_parameters():
                        if 'threshold' in name:
                            param.clamp_(0.1, 3.0)
            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())
            
            # Explicit cleanup to prevent memory accumulation
            if phase in [1, '1a', '1b'] and use_supmoco:
                del q_inputs, k_inputs, key_features, queue_features, queue_labels, queue_ages, q_video_ids, q_timestamps
            else:
                del inputs
            del targets_c, outputs, loss

        # Periodic memory cleanup after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        avg_train_loss = train_loss / len(train_loader)
          
        val_loss, val_acc, val_bal_acc, val_dice, val_iou, val_pre, val_rec = validate(model, val_loader, loss_fn, device, only_classification=phase in [1, '1a', '1b'])
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # --- Offline Probes (Phase 1A only) ---
        val_linear_acc = 0.0
        train_linear_acc = 0.0
        subj_cv_acc = 0.0
        subj_cross_acc = 0.0
        if phase == '1a' and (epoch % probe_interval == 0 or epoch == epochs - 1):
            from test import _linear_probe_cv, _linear_probe_accuracy, _svm_probe_cv, _svm_probe_accuracy
            
            # Temporary inline extractor
            def _extract_feats(loader):
                embs, lbls, grps = [], [], []
                model.eval()
                with torch.no_grad():
                    for batch in loader:
                        vid = batch[0].to(device)
                        if len(batch) >= 8: # Augmented train batch
                            lbl = batch[3].to(device)
                            bag_id = batch[5]
                        else: # Standard val batch
                            lbl = batch[2].to(device)
                            bag_id = batch[3]
                        
                        K_bag = None
                        if vid.dim() == 6:
                            B, K_bag, T, C, H, W = vid.shape
                            vid = vid.view(B * K_bag, T, C, H, W)
                        else:
                            B, T, C, H, W = vid.shape

                        vid = vid.permute(1, 0, 2, 3, 4)
                        features, _ = model.encoder(vid)

                        if features.dim() == 5:
                            emb = features.mean(dim=[0, 3, 4])
                        else:
                            emb = features.mean(dim=[-2, -1])

                        if K_bag is not None:
                            emb = emb.view(B, K_bag, -1).mean(dim=1)

                        emb = F.normalize(emb, dim=1, eps=1e-6)
                        embs.append(emb.cpu())
                        lbls.append(lbl.cpu())
                        grps.extend(bag_id)
                        
                return torch.cat(embs, dim=0).numpy(), torch.cat(lbls, dim=0).numpy(), np.array([int(b) for b in grps])
            
            # Need to get subjects from dataset
            tr_b2s = {str(item[0]): item[3] for item in train_loader.dataset.samples}
            vl_b2s = {str(item[0]): item[3] for item in val_loader.dataset.samples}
            
            emb_t, lbl_t, grps_t = _extract_feats(train_loader)
            emb_v, lbl_v, grps_v = _extract_feats(val_loader)
            
            subj_t = np.array([tr_b2s.get(str(g), -1) for g in grps_t])
            subj_v = np.array([vl_b2s.get(str(g), -1) for g in grps_v])
            
            train_linear_acc = _linear_probe_cv(emb_t, lbl_t, grps_t, n_folds=5)
            val_linear_acc = _linear_probe_accuracy(emb_t, lbl_t, emb_v, lbl_v)
            
            # Proxy-A
            if not np.all(subj_t == -1):
                subj_cv_acc = _svm_probe_cv(emb_t, subj_t, grps_t, n_folds=5)
                subj_cross_acc = _svm_probe_accuracy(emb_t, subj_t, emb_v, subj_v)
            
            print(f"  Offline Probes — Emotion Train CV: {train_linear_acc:.4f} | Val Bal Acc: {val_linear_acc:.4f}")
            print(f"  Proxy-A Probes — Subject Train CV (SVM): {subj_cv_acc:.4f} | Val Cross Acc (SVM): {subj_cross_acc:.4f}")

        # --- DANN Logging ---
        if getattr(model, 'use_dann', False):
            train_dann_acc = train_dann_correct / max(train_dann_total, 1)
            avg_dann_loss = dann_loss_total / max(train_dann_total, 1)
            print(f"  DANN — Train Subject Accuracy: {train_dann_acc:.4f} | Loss: {avg_dann_loss:.4f}")
            log_dict[f'Phase{str(phase).upper()}/Train/DANN_Subject_Acc'] = train_dann_acc
            log_dict[f'Phase{str(phase).upper()}/Train/DANN_Loss'] = avg_dann_loss



        # --- Checkpoint: checkpoint_last.pt (every epoch, atomic write) ---
        save_checkpoint(model, optimizer, scheduler, epoch, best_acc, best_dice,
                        f"{checkpoint_dir}/checkpoint_temp.pt",
                        momentum_model=momentum_model, supmoco_state=supmoco_state) 
        os.replace(f"{checkpoint_dir}/checkpoint_temp.pt", f"{checkpoint_dir}/checkpoint_last.pt")
        
        if phase in [1, '1a', '1b']:
            print(f"Phase {phase} Epoch {epoch} | LR: {current_lr:.2e} | "
                  f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f}")
        else:
            print(f"Phase {phase} Epoch {epoch} | LR: {current_lr:.2e} | "
                  f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | Dice: {val_dice:.4f} | "
                  f"Val Pre: {val_pre:.4f} | Val Rec: {val_rec:.4f}")
        
        log_dict = {
            f"Phase{phase}/Train/Loss": avg_train_loss,
            f"Phase{phase}/Val/Loss": val_loss,
            f"Phase{phase}/Val/Accuracy": val_acc,
            f"Phase{phase}/Val/Balanced_Accuracy": val_bal_acc,
            "LR": current_lr
        }
        if phase in [1, '1a', '1b'] and use_supmoco and supmoco_state is not None:
            queue_len = int(supmoco_state.queue_filled.item())
            log_dict.update({
                f"Phase{phase}/Train/QueueSize": queue_len,
                f"Phase{phase}/Train/QueueFillRatio": queue_len / float(max(1, supmoco_state.queue_size)),
            })
        if phase == '1a' and (epoch % probe_interval == 0 or epoch == epochs - 1):
            log_dict[f"Phase{phase}/Train/Emotion_Probe_CV_Acc"] = train_linear_acc
            log_dict[f"Phase{phase}/Val/Emotion_Probe_Val_Acc"] = val_linear_acc
            log_dict[f"Phase{phase}/Train/Subject_Probe_CV_Acc"] = subj_cv_acc
            log_dict[f"Phase{phase}/Val/Subject_Probe_Val_Acc"] = subj_cross_acc
        if phase not in [1, '1a', '1b']:
            log_dict.update({
                f"Phase{phase}/Val/Dice": val_dice,
                f"Phase{phase}/Val/IoU": val_iou,
                f"Phase{phase}/Val/Precision": val_pre,
                f"Phase{phase}/Val/Recall": val_rec,
            })

        for k, v in log_dict.items(): writer.add_scalar(k, v, epoch)

        # --- Phase 1A: Three-tier checkpointing ---
        if phase == '1a':
            # checkpoint_best.pt: save when train loss decreases
            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                best_acc = avg_train_loss  # Store loss in acc field for Phase 1A
                save_checkpoint(model, optimizer, scheduler, epoch, best_acc, best_dice,
                                f"{checkpoint_dir}/checkpoint_temp.pt",
                                momentum_model=momentum_model, supmoco_state=supmoco_state)
                os.replace(f"{checkpoint_dir}/checkpoint_temp.pt", f"{checkpoint_dir}/checkpoint_best.pt")
                wandb.save(f"{checkpoint_dir}/checkpoint_best.pt")
            else:
                print(f"Best Train Loss yet: {best_train_loss:.4f}")

            # checkpoint_best_linear.pt: save when offline linear eval accuracy increases
            if val_linear_acc > best_linear_acc:
                best_linear_acc = val_linear_acc
                save_checkpoint(model, optimizer, scheduler, epoch, best_acc, best_dice,
                                f"{checkpoint_dir}/checkpoint_temp.pt",
                                momentum_model=momentum_model, supmoco_state=supmoco_state)
                os.replace(f"{checkpoint_dir}/checkpoint_temp.pt", f"{checkpoint_dir}/checkpoint_best_linear.pt")
                wandb.save(f"{checkpoint_dir}/checkpoint_best_linear.pt")
                print(f"  New best Linear Eval Acc: {best_linear_acc:.4f}")
        else:
            # Phase 2+: original acc/dice logic
            if val_acc > best_acc:
                best_acc = val_acc
                best_dice = val_dice
                save_checkpoint(model, optimizer, scheduler, epoch, best_acc, best_dice,
                                f"{checkpoint_dir}/checkpoint_temp.pt")
                os.replace(f"{checkpoint_dir}/checkpoint_temp.pt", f"{checkpoint_dir}/checkpoint_best.pt")
                wandb.save(f"{checkpoint_dir}/checkpoint_best.pt")
            elif val_acc == best_acc and val_dice > best_dice:
                best_dice = val_dice
                save_checkpoint(model, optimizer, scheduler, epoch, best_acc, best_dice,
                                f"{checkpoint_dir}/checkpoint_{epoch:03d}_{best_acc:.4f}_{best_dice:.4f}.pt")
            else:
                print(f"Best Result yet: {best_acc:.4f}")

        wandb.log(log_dict, step=epoch)


def phase_one_a(config, model, device, train_loader, val_loader, writer, checkpoint_dir, resume, checkpoint=None):
    print("=== Phase 1A: Training Encoder Only (Old Style, No MIL/Bagging, No SwiGLU) ===")
    
    if config.training.get('bagging', False):
        print("WARNING: 'bagging' is set to True in config, but Phase 1A enforces old-style 1s windows. Ensure dataset complies!")

    unique_subjects = sorted(list(set([s[3] for s in train_loader.dataset.samples if s[3] != -1])))
    num_dynamic_subjects = max(len(unique_subjects), 1)
    subj_remapper = {actual_id: mapped_id for mapped_id, actual_id in enumerate(unique_subjects)}
    print(f"Dynamic DANN Subject Count: {num_dynamic_subjects} (Train Split)")

    if isinstance(model, SpikingMobileNetProjector):
        enc_class = model
        # Try to forcefully disable SwiGLU if it was constructed with it
        if hasattr(enc_class, 'use_swiglu') and enc_class.use_swiglu:
             print("WARNING: Model initialized with SwiGLU, but Phase 1A enforces use_swiglu=False. Overriding where possible.")
    else:
        enc_class = SpikingMobileNetProjector(
            encoder_backbone = model.encoder,
            num_classes=config.model.get('num_classes', 5),
            use_swiglu=False,  # FORCE FALSE
            use_dann=config.loss.get('use_dann', False),
            num_subjects=num_dynamic_subjects
        ).to(device)

    if getattr(enc_class, 'use_dann', False) and not hasattr(enc_class, 'dann_head'):
        # If the model was pre-instantiated but somehow doesn't have the DANN head, log a warning
        print("WARNING: use_dann is True but model has no dann_head. Ignoring DANN for this phase.")
        enc_class.use_dann = False

    iic_enabled = config.loss.get('iic_enabled', False)
    iic_intra_weight = config.loss.get('iic_intra_weight', 1.0)
    iic_inter_weight = config.loss.get('iic_inter_weight', 1.0)
    decoupled = config.loss.get('decoupled', False)
    con_temp = config.loss.get('temperature', 0.07)
    use_supmoco = config.training.get('use_supmoco', False)

    if use_supmoco:
        loss_fn = SupMoCoLoss(
            train_loader.dataset.prototypes,
            temperature=con_temp,
            iic_enabled=iic_enabled,
            iic_intra_weight=iic_intra_weight,
            iic_inter_weight=iic_inter_weight,
            temporal_decay_enabled=config.training.get('temporal_decay_enabled', False),
            temporal_decay_factor=config.training.get('temporal_decay_factor', 0.999),
            exclude_same_trial=config.loss.get('exclude_same_trial', True)
        )
    else:
        loss_fn = ContrastiveLoss(
            train_loader.dataset.prototypes,
            temperature=con_temp,
            iic_enabled=iic_enabled,
            iic_intra_weight=iic_intra_weight,
            iic_inter_weight=iic_inter_weight,
            decoupled=decoupled,
        )

    loss_fn.to(device)

    start_epoch = 0
    best_acc = 100.0
    best_dice = 0.0
    epochs = config.training.phase_1_epochs
    warmup_epochs = config.training.get('warmup_epochs', 0)
    accumulation_steps = config.training.get('accumulation_steps', 1)
    
    optimizer = create_optimizer(enc_class, loss_fn, config, low_encoder_lr=False)
    scheduler = create_hybrid_scheduler(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)

    # Load model weights FIRST (before building momentum encoder)
    if resume and checkpoint is not None:
        enc_class.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['accuracy']
        best_dice = checkpoint['dice']
        print(f"Resuming training from epoch {start_epoch}...")

    # Build momentum encoder AFTER loading weights so it inherits restored state
    momentum_model = None
    supmoco_state = None
    if use_supmoco:
        momentum_model = build_momentum_encoder(enc_class).to(device)
        supmoco_state = SupMoCoState(
            queue_size=config.training.get('supmoco_queue_size', 4096),
            feature_dim=config.training.get('supmoco_feature_dim', 128),
        ).to(device)
        # Restore saved momentum/queue state if available
        if resume and checkpoint is not None:
            if checkpoint.get('momentum_model_state_dict') is not None:
                momentum_model.load_state_dict(checkpoint['momentum_model_state_dict'])
                print("  Restored momentum encoder weights from checkpoint.")
            if checkpoint.get('supmoco_state_dict') is not None:
                supmoco_state.load_state_dict(checkpoint['supmoco_state_dict'], strict=False)
                print(f"  Restored SupMoCo queue (filled={int(supmoco_state.queue_filled.item())}).")



    training_loop(
        '1a',
        start_epoch,
        epochs,
        best_acc,
        best_dice,
        enc_class,
        device,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        scheduler,
        writer,
        checkpoint_dir,
        use_supmoco=use_supmoco,
        moco_momentum=config.training.get('supmoco_momentum', 0.999),
        momentum_model=momentum_model,
        supmoco_state=supmoco_state,
        unfreeze_epoch=-1,
        accumulation_steps=accumulation_steps,
        dann_weight=config.loss.get('dann_weight', 1.0),
        dann_alpha=config.loss.get('dann_alpha', 1.0),
        subj_remapper=subj_remapper,
        use_mmd=config.loss.get('use_mmd', False),
        lambda_mmd=config.loss.get('lambda_mmd', 0.0)
    )
   

def phase_one_b(config, model, device, train_loader, val_loader, writer, checkpoint_dir, resume, checkpoint=None):
    print("=== Phase 1B: Training Encoder with MIL/Bagging & Arch Upgrades ===")
    
    unique_subjects = sorted(list(set([s[3] for s in train_loader.dataset.samples if s[3] != -1])))
    num_dynamic_subjects = max(len(unique_subjects), 1)
    subj_remapper = {actual_id: mapped_id for mapped_id, actual_id in enumerate(unique_subjects)}
    print(f"Dynamic DANN Subject Count: {num_dynamic_subjects} (Train Split)")

    if isinstance(model, SpikingMobileNetProjector):
        enc_class = model
    else:
        enc_class = SpikingMobileNetProjector(
            encoder_backbone = model.encoder,
            num_classes=config.model.get('num_classes', 5),
            use_swiglu=config.model.get('use_swiglu', True),
            use_dann=config.loss.get('use_dann', False),
            num_subjects=num_dynamic_subjects
        ).to(device)
        
    if getattr(enc_class, 'use_dann', False) and not hasattr(enc_class, 'dann_head'):
        print("WARNING: use_dann is True but model has no dann_head. Ignoring DANN for this phase.")
        enc_class.use_dann = False

    iic_enabled = config.loss.get('iic_enabled', False)
    iic_intra_weight = config.loss.get('iic_intra_weight', 1.0)
    iic_inter_weight = config.loss.get('iic_inter_weight', 1.0)
    decoupled = config.loss.get('decoupled', False)
    con_temp = config.loss.get('temperature', 0.07)
    use_supmoco = config.training.get('use_supmoco', False)

    if use_supmoco:
        loss_fn = SupMoCoLoss(
            train_loader.dataset.prototypes,
            temperature=con_temp,
            iic_enabled=iic_enabled,
            iic_intra_weight=iic_intra_weight,
            iic_inter_weight=iic_inter_weight,
            temporal_decay_enabled=config.training.get('temporal_decay_enabled', False),
            temporal_decay_factor=config.training.get('temporal_decay_factor', 0.999),
            exclude_same_trial=config.loss.get('exclude_same_trial', True)
        )
    else:
        loss_fn = ContrastiveLoss(
            train_loader.dataset.prototypes,
            temperature=con_temp,
            iic_enabled=iic_enabled,
            iic_intra_weight=iic_intra_weight,
            iic_inter_weight=iic_inter_weight,
            decoupled=decoupled,
        )

    loss_fn.to(device)

    start_epoch = 0
    best_acc = 100.0
    best_dice = 0.0
    epochs = config.training.phase_1_epochs
    warmup_epochs = config.training.get('warmup_epochs', 0)
    accumulation_steps = config.training.get('accumulation_steps', 1)
    
    unfreeze_epoch = config.training.get('freeze_encoder_epochs', 0)
    
    # In 1B, if resuming, normal resume.
    # If not resuming but checkpoint provided, it's a backbone transfer!
    if resume and checkpoint is not None:
        enc_class.load_state_dict(checkpoint['model_state_dict'])
        # Optimizer and scheduler loads handled later...
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['accuracy']
        best_dice = checkpoint['dice']
        print(f"Resuming training from epoch {start_epoch}...")
    elif not resume and checkpoint is not None:
        print("Transferring pretrained weights from checkpoint as Backbone...")
        enc_class.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Initialize optimizer WITH requires_grad=True across the board so that
    # the optimizer recognizes the params.
    for param in enc_class.parameters():
        param.requires_grad = True

    # Note: lower encoder LR
    optimizer = create_optimizer(enc_class, loss_fn, config, low_encoder_lr=True)
    scheduler = create_hybrid_scheduler(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)

    momentum_model = None
    supmoco_state = None
    if use_supmoco:
        momentum_model = build_momentum_encoder(enc_class).to(device)
        supmoco_state = SupMoCoState(
            queue_size=config.training.get('supmoco_queue_size', 4096),
            feature_dim=config.training.get('supmoco_feature_dim', 128),
        ).to(device)
    
    if resume and checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    training_loop(
        '1b',
        start_epoch,
        epochs,
        best_acc,
        best_dice,
        enc_class,
        device,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        scheduler,
        writer,
        checkpoint_dir,
        use_supmoco=use_supmoco,
        moco_momentum=config.training.get('supmoco_momentum', 0.999),
        momentum_model=momentum_model,
        supmoco_state=supmoco_state,
        unfreeze_epoch=unfreeze_epoch,
        accumulation_steps=accumulation_steps,
        dann_weight=config.loss.get('dann_weight', 1.0),
        dann_alpha=config.loss.get('dann_alpha', 1.0),
        subj_remapper=subj_remapper,
        use_mmd=config.loss.get('use_mmd', False),
        lambda_mmd=config.loss.get('lambda_mmd', 0.0)
    )

    
def phase_two(config, model, device, train_loader, val_loader, writer, checkpoint_dir, resume, checkpoint):
    print("=== Phase Two: Training Rest Only ===")
    initialize_network(model, train_loader, device)
    loss_fn = FullHybridLoss(
        smooth = 0.0,
        lambda_seg = config.loss.get('lambda_seg', 1.0),
        lambda_con = 0.0,
        lambda_class = config.loss.get('lambda_class', 1.0),
        alpha = config.loss.get('alpha', 0.5),
        beta = config.loss.get('beta', 0.5),
        time_steps=config.data.get('num_timesteps', 16),
    )
    loss_fn.class_loss.masks = train_loader.dataset.prototypes

    loss_fn.to(device)
    if not resume:
        new_dict = {k:v for k,v in checkpoint['model_state_dict'].items() if 'encoder' in k}
        model.load_state_dict(new_dict, strict=False)

    start_epoch = 0
    best_acc = 0.0
    best_dice = 0.0
    epochs = config.training.phase_2_epochs
    accumulation_steps = config.training.get('accumulation_steps', 1)

    for param in model.encoder.parameters():
        param.requires_grad = False

    optimizer = create_optimizer(model, loss_fn, config, low_encoder_lr=False)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    

    if resume and checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['accuracy']
        best_dice = checkpoint['dice']
        print(f"Resuming training from epoch {start_epoch}...")
    
    training_loop('2', start_epoch, epochs, best_acc, best_dice, model, device, train_loader, val_loader, loss_fn, optimizer, scheduler, writer, checkpoint_dir, unfreeze_epoch=-1, accumulation_steps=accumulation_steps)


def phase_three(config, model, device, train_loader, val_loader, writer, checkpoint_dir, resume, checkpoint):  
    print("=== Phase Three: Training Model ===")
    
    loss_fn = FullHybridLoss(
        smooth = 0.0,
        lambda_seg = config.loss.get('lambda_seg', 1.0),
        lambda_con = 0.0,
        lambda_class = config.loss.get('lambda_class', 1.0),
        alpha = config.loss.get('alpha', 0.5),
        beta = config.loss.get('beta', 0.5),
        time_steps=config.data.get('num_timesteps', 16),
    )

    loss_fn.add_fire_rate_loss(model, 
                               lambda_fire=config.loss.get('lambda_fire', 0.1), 
                               target_rate=config.loss.get('target_rate', 0.05))

    loss_fn.to(device)

    if not resume:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    for param in model.parameters():
        param.requires_grad = False
    
    for name, param in model.named_parameters():
        if "encoder" not in name:
            param.requires_grad = True
    
    for param in model.encoder.layer4a.parameters():
        param.requires_grad = True
    
    for param in model.encoder.layer4b.parameters():
        param.requires_grad = True

    model.encoder.apply(freeze_bn_stats)

    start_epoch = 0
    best_acc = 0.0
    best_dice = 0.0
    epochs = config.training.phase_3_epochs
    accumulation_steps = config.training.get('accumulation_steps', 1)
    
    optimizer = create_optimizer(model, loss_fn, config, low_encoder_lr=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    if resume and checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['accuracy']
        best_dice = checkpoint['dice']
        print(f"Resuming training from epoch {start_epoch}...")
    
    training_loop('3', start_epoch, epochs, best_acc, best_dice, model, device, train_loader, val_loader, loss_fn, optimizer, scheduler, writer, checkpoint_dir, unfreeze_epoch=-1, accumulation_steps=accumulation_steps)

    
phase_handles = {
    '1a': phase_one_a,
    '1b': phase_one_b,
    '1': phase_one_a, # fallback for configs specifying '1'
    '2': phase_two,
    '3': phase_three,
}


def run_training(config, model, device, phase, resume, loso=None, subj=None, checkpoint=None):

    if loso:
        run_name = f"{config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_loso{loso}"
    else:
        run_name = f"{config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_subj{subj}"
    checkpoint_dir = os.path.join(config.data.get('save_path', ''), "saved_models", f"phase{phase}", run_name)
    
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    wandb.init(
        project=config.logging.project_name,
        name=config.logging.run_name,
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        tags=list(config.logging.tags),
        mode="disabled" if config.logging.get('offline') else "online",
        settings=wandb.Settings(_disable_stats=True, _disable_meta=True) 
    )
    
    log_dir = os.path.join("results", run_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"Initializing TensorBoard: {log_dir}")

    masks = torch.load(os.path.join(config.data.dataset_path,'masks.pt')).to(device)
    density = masks.sum() / masks.numel()
    print(f"Global Density (Consider this for setting up target firing rate): {density:.4f}")

    train_aug = nn.Sequential(
        GaussianNoise(std=0.05),
        FrequencyDropout(p=0.1), # Kills spectral biometric (Alpha peak)
        VideoRandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3)), # Kills spatial biometric (Skull/Electrodes)
        #SpatialDropout(p=0.3), # Kills physical cap impedance variations (Specific point electrodes)
        VideoTemporalMasking(p=0.1, max_mask_len=8), # Kills temporal biometric (ODConv barcode)
        SignalJitter(lower=0.5, upper=2.0) # Kills absolute power biometric (Impedance)
    )

    train_set = SWEEPDataset(
        config,
        split='train',
        #experiment=True,
        loso=loso,
        subj=subj,
        prototypes=masks,
        augmentations=train_aug
    )
    
    val_set = SWEEPDataset(
        config, 
        split='val',
        #experiment=True,
        loso=loso,
        subj=subj,
        prototypes=masks
    )

    num_workers = config.data.get('num_workers', 0)
    prefetch = config.data.get('prefetch_factor', 2) if num_workers > 0 else None
    persist = num_workers > 0  # Only use persistent_workers if num_workers > 0
    
    use_pk_sampler = config.training.get('use_pk_sampler', True)
    if use_pk_sampler:
        train_loader = DataLoader(train_set, 
                                  batch_sampler=PKSampler(train_set, 
                                                          batch_size=config.training.batch_size, 
                                                          n_classes=config.model.get('n_emotions', 5),
                                                          subject_diverse_k=config.training.get('subject_diverse_k', True)),
                                  num_workers=num_workers,
                                  prefetch_factor=prefetch,
                                  persistent_workers=persist,
                                  pin_memory=True)
    else:
        train_loader = DataLoader(train_set, 
                                  batch_size=config.training.batch_size,
                                  shuffle=True, 
                                  num_workers=num_workers,
                                  prefetch_factor=prefetch,
                                  persistent_workers=persist,
                                  pin_memory=True)
    
    val_loader = DataLoader(val_set, 
                            batch_size=config.training.batch_size, 
                            shuffle=False, 
                            num_workers=num_workers,
                            prefetch_factor=prefetch,
                            persistent_workers=persist,
                            pin_memory=True)

    print(f"Data Loaded: {len(train_set)} Train | {len(val_set)} Val")
    
    model = model.to(device)
    phase_handles[phase](config, model, device, train_loader, val_loader, writer, checkpoint_dir, resume, checkpoint)
    
    print("--- Training Complete ---")
    wandb.finish()
    writer.close()