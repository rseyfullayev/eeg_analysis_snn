# CITE https://github.com/HobbitLong/SupContrast/tree/master

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.snn_modeling.utils.loss import ContrastiveLoss, SupMoCoLoss
from src.snn_modeling.utils.optim import Muon, HybridOptimizer, HybridScheduler
from src.snn_modeling.utils.nan_monitor import NaNMonitor
from src.snn_modeling.utils.live_tracker import LiveTracker
from src.snn_modeling.dataloader.dataset import SWEEPDataset, PKSampler
from src.snn_modeling.utils.supmoco import SupMoCoState, build_momentum_encoder, momentum_update
from src.snn_modeling.utils.sm_loss import SMCLossEngine
from src.snn_modeling.utils.certification import alpha_identity_escape
import os
import math
import gc
from datetime import datetime
from sklearn.metrics import balanced_accuracy_score
import segmentation_models_pytorch as smp
import wandb
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import numpy as np
from tqdm import tqdm
from src.snn_modeling.utils.augmentations import VideoTemporalMasking, GaussianNoise, FrequencyDropout, SignalJitter, VideoRandomErasing, SpatialDropout, FourierMixup, build_fourier_mixup
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
                    momentum_model=None, supmoco_state=None, smc_engine=None, best_metric=None):
    payload = {
        'best_metric': best_metric,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy': acc,
        'dice': dice,
        'momentum_model_state_dict': momentum_model.state_dict() if momentum_model is not None else None,
        'supmoco_state_dict': supmoco_state.state_dict() if supmoco_state is not None else None,
        'sm_state_dict': smc_engine.state_dict() if smc_engine is not None else None,
        'sm_state': smc_engine.sm_state() if smc_engine is not None else None,
    }
    torch.save(payload, path)
    print(f"Saved checkpoint at Epoch {epoch} (Acc: {acc:.4f}, Dice: {dice:.4f})")


def validate(model, val_loader, criterion, device, threshold=0.5, only_classification=False):
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
            # Standard unpacking
            inputs, targets, labels = batch[0], batch[1], batch[2]
            inputs, targets, labels = inputs.to(device), targets.to(device), labels.to(device)
            
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

            if only_classification:
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                loss = criterion(logits, labels) if not isinstance(criterion, torch.nn.CrossEntropyLoss) else criterion(logits, labels.squeeze())
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                val_loss += loss.item()
                val_loop.set_postfix(loss=loss.item())
                continue

            loss = criterion(outputs, targets, labels)
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
                
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
                
    avg_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
    accuracy = correct / total if total > 0 else 0
    try:
        balanced_acc = balanced_accuracy_score(all_targets, all_preds)
    except:
        balanced_acc = 0

    if only_classification:
        return avg_loss, accuracy, balanced_acc, 0, 0, 0, 0

    eps = 1e-7
    dice_score = (2 * tp_tot) / (2 * tp_tot + fp_tot + fn_tot + eps)
    iou_score = tp_tot / (tp_tot + fp_tot + fn_tot + eps)
    precision = tp_tot / (tp_tot + fp_tot + eps)
    recall = tp_tot / (tp_tot + fn_tot + eps)

    return avg_loss, accuracy, balanced_acc, dice_score, iou_score, precision, recall

def create_optimizer(model, loss_fn, config, low_encoder_lr=False):
    lr = config.training.get('learning_rate', 1e-3)
    use_muon = config.training.get('use_muon', False)
    weight_decay = config.training.get('weight_decay', 1e-4)

    # ── Identify special (no-decay) parameter IDs ──
    no_decay_id = set()
    classess_names = (ALIF,
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
    # NOTE: Always keep all 5 groups even if empty, for checkpoint compatibility
    # with legacy (pre-Muon) state dicts that always had 5 param groups.
    adam_groups = [
        {'params': adam_base_params, 'lr': lr, 'weight_decay': weight_decay},
        {'params': adam_encoder_params, 'lr': lr * 1e-2, 'weight_decay': weight_decay},
        {'params': adam_no_decay_params, 'lr': lr, 'weight_decay': 0.0},
        {'params': time_params, 'lr': lr * 0.5, 'weight_decay': 0.0},
        {'params': threshold_params, 'lr': lr * 1.0, 'weight_decay': 0.0},
    ]
    optimizer_adamw = optim.AdamW(adam_groups, betas=(0.9, 0.999))

    # ── Build Muon (if requested) — uses torch.optim.Muon (PyTorch >= 2.9) ──
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
            print(f"\n=== Muon+AdamW Hybrid (torch.optim.Muon) ===")
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
                  use_supmoco=False,
                  moco_momentum=0.999,
                  momentum_model=None,
                  supmoco_state=None,
                  smc_engine=None,
                  unfreeze_epoch=-1,
                  accumulation_steps=1,
                  best_train_loss=float('inf'),
                  probe_interval=5,
                  live_tracker=None):

    # Phase 1A trackers (best_train_loss injectable for honest resume — B15)
    best_linear_acc = 0.0

    # NaN/Inf monitor with diagnostic reasoning
    if live_tracker is None:
        live_tracker = LiveTracker()  # disabled fallback (no credentials)
    nan_monitor = NaNMonitor(model, verbose=True, max_reports_per_epoch=5, live_tracker=live_tracker)

    device_info = str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "CPU"
    live_tracker.training_started(phase, epochs, device_info=device_info)

    # B3(iii) early-stop series (§11.5 / certification.alpha_identity_escape):
    # per-epoch intra-trial Stdt(alpha) + anneal-floor flags (SMC runs only).
    smc_std_series, smc_anneal_series = [], []

    for epoch in range(start_epoch, epochs):
        model.train()
        loss_fn.current_epoch = epoch
        loss_fn.step_count = 0
        nan_monitor.reset_epoch()

        if unfreeze_epoch > 0:
            if epoch < unfreeze_epoch:
                for param in model.encoder.parameters():
                    param.requires_grad = False
            elif epoch == unfreeze_epoch:
                print("--- Unfreezing Encoder ---")
                for param in model.encoder.parameters():
                    param.requires_grad = True

        train_loss = 0.0
        train_emo_correct = 0
        train_emo_total = 0
        last_smc_diag = None

        train_loop = tqdm(train_loader, desc=f"Phase {phase} Epoch {epoch+1}/{epochs}", unit="batch")
        for batch_idx, batch in enumerate(train_loop):
            if live_tracker and getattr(live_tracker, 'is_cancelled', False):
                print("\n[!] Run cancelled via Telegram Dashboard. Exiting...")
                live_tracker.custom("🛑 **Run manually cancelled by user via Dashboard.**")
                import sys
                sys.exit(0)
                
            # Unpack batch: 8-tuple = dual-view augmented (SupMoCo), 6-tuple = single-view.
            if len(batch) == 8:
                inp1, inp2, targets, targets_c, subject_labels, _, video_ids, timestamps = batch
                video_ids = video_ids.to(device)
                timestamps = timestamps.to(device)
                inp1, inp2 = inp1.to(device), inp2.to(device)
                subject_labels = subject_labels.to(device)
                if use_supmoco:
                    q_inputs = inp1
                    k_inputs = inp2
                else:
                    inputs = torch.cat([inp1, inp2], dim=0)
            else:
                inputs, targets, targets_c, _, video_ids, timestamps = batch
                video_ids = video_ids.to(device)
                timestamps = timestamps.to(device)
                inputs = inputs.to(device)
                if use_supmoco:
                    raise ValueError("SupMoCo requires dual-view augmented batches (len(batch) == 8).")
                
            targets, targets_c = targets.to(device), targets_c.to(device)

            if use_supmoco:
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

                # Monitor: check activations after forward
                nan_monitor.check_activations(epoch=epoch, batch_idx=batch_idx,
                                              backbone_feats=backbone_feats, outputs=outputs)

                with torch.no_grad():
                    if smc_engine is not None:
                        # §11.4i: the 128-D ProjectionHead leaves the SMC queue/anchor path —
                        # keys must come from the momentum BACKBONE (256-D), matching z_q.
                        key_features = momentum_model.extract_features(k_inputs, K=K_bag)
                    else:
                        key_features = momentum_model(k_inputs, K=K_bag)
                    if K_bag is not None and key_features.shape[0] == B * K_bag:
                        key_features = key_features.view(B, K_bag, *key_features.shape[1:]).mean(dim=1)
                    key_features = F.normalize(key_features, dim=1, eps=1e-6)

                if smc_engine is not None:
                    # Eq. 23 engine branch (plan §5 Stage 6): online z_q vs momentum key_z;
                    # queue reads, E-step, anchor/OT/gate/lambda and all M-step side
                    # effects (rho, geometry, enqueue) live inside SMCLossEngine.forward().
                    z_q = F.normalize(backbone_feats, dim=1, eps=1e-6)
                    wins = timestamps.reshape(-1) if timestamps.dim() == 1 else timestamps[:, 0]
                    smc_out = smc_engine(z_q, key_features, targets_c, subject_labels,
                                         wins, video_ids)
                    loss = smc_out['total']
                    last_smc_diag = smc_out['diagnostics']
                else:
                    queue_features, queue_labels, queue_subject_labels, queue_ages, q_video_ids, q_timestamps = supmoco_state.get_queue()
                    loss = loss_fn(outputs, key_features, targets_c, subject_labels,
                                   queue_features, queue_labels, queue_subject_labels, queue_ages,
                                   query_video_ids=video_ids, query_timestamps=timestamps,
                                   queue_video_ids=q_video_ids, queue_timestamps=q_timestamps)

                # Monitor: check loss
                nan_monitor.check_loss(epoch=epoch, batch_idx=batch_idx, contrastive_loss=loss)

                # --- Offline probes run during validation (below). DANN/MMD/VIB
                # adversarial paths were retired in Stage 0 tranche 2 (fixed.pdf §13).


            else:
                # Legacy dual-view ContrastiveLoss (no queue): inputs already carry
                # the concatenated views from the unpack stage.
                K_bag = None
                if inputs.dim() == 6:
                    B, K_bag, T, C, H, W = inputs.shape
                    inputs = inputs.view(B * K_bag, T, C, H, W)
                else:
                    B, T, C, H, W = inputs.shape

                # SNN requires Time to be dimension 0: [T, Batch, C, H, W]
                inputs = inputs.permute(1, 0, 2, 3, 4)

                outputs = model(inputs, K=K_bag)

                # If the model didn't internally reduce K, average bag windows
                if K_bag is not None and outputs.shape[0] == B * K_bag:
                    outputs = outputs.view(B, K_bag, *outputs.shape[1:]).mean(dim=1)

                if B % 2 != 0:
                    raise ValueError(
                        f"No-queue contrastive path needs an even concatenated batch (got {B}); "
                        "set training.use_supmoco=true (the supported configuration).")
                f1, f2 = torch.split(outputs, [B // 2, B // 2], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                loss = loss_fn(features, targets_c)
           
            loss = loss / accumulation_steps
            loss.backward()
            
            # Interactive bot state update (every 10 batches to avoid excessive IO)
            if batch_idx % 10 == 0:
                kwargs = {}
                if hasattr(loss_fn, 'ema_pos_dots'):
                    kwargs['EMA Pos Dot'] = loss_fn.ema_pos_dots
                    kwargs['EMA Neg Dot'] = loss_fn.ema_neg_dots
                live_tracker.update_batch_state(epoch + 1, batch_idx + 1, len(train_loader), loss.item() * accumulation_steps, **kwargs)
            
            is_step = ((batch_idx + 1) % accumulation_steps == 0) or ((batch_idx + 1) == len(train_loader))

            if is_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Monitor: check gradients after clipping but before optimizer step
                nan_monitor.check_gradients(epoch=epoch, batch_idx=batch_idx)

                optimizer.step()

                # Monitor: check weights after optimizer step
                nan_monitor.check_weights(epoch=epoch, batch_idx=batch_idx)

                if use_supmoco:
                    momentum_update(model, momentum_model, moco_momentum)
                optimizer.zero_grad(set_to_none=True)

            if use_supmoco and smc_engine is None:
                with torch.no_grad():
                    supmoco_state.enqueue(key_features, targets_c, subject_labels,
                                          video_ids=video_ids, timestamps=timestamps)

            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())
            
            # Explicit cleanup to prevent memory accumulation
            if smc_engine is not None:
                del q_inputs, k_inputs, key_features, z_q, smc_out
            elif use_supmoco:
                del q_inputs, k_inputs, key_features, queue_features, queue_labels, queue_ages, q_video_ids, q_timestamps
            else:
                del inputs
            del targets_c, loss
            if 'outputs' in locals():
                del outputs

        # Periodic memory cleanup after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # NaN Monitor epoch summary
        nan_monitor.summary()

        avg_train_loss = train_loss / len(train_loader)

        val_loss, val_acc, val_bal_acc, val_dice, val_iou, val_pre, val_rec = validate(
            model, val_loader, loss_fn, device, only_classification=True)
            
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # --- Live Tracker: epoch summary ---
        epoch_metrics = {
            "train_loss": avg_train_loss,
            "train_acc": train_emo_correct / max(train_emo_total, 1),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_bal_acc": val_bal_acc,
            "val_dice": val_dice,
            "lr": current_lr,
        }
        # --- WeightWatcher (Empirical Generalization Analysis) ---
        if epoch % probe_interval == 0 or epoch == epochs - 1:
            try:
                import weightwatcher as ww
                import logging
                ww_logger = logging.getLogger("weightwatcher")
                ww_logger.setLevel(logging.CRITICAL)  # Suppress massive spam
                
                # 1. Analyze frozen backbone
                watcher_enc = ww.WeightWatcher(model=model.encoder)
                details_enc = watcher_enc.analyze()
                if details_enc is not None and not details_enc.empty and 'alpha' in details_enc.columns:
                    enc_alpha = details_enc['alpha'].mean()
                    epoch_metrics["ww_alpha_encoder"] = enc_alpha
                else:
                    enc_alpha = 0.0
                
                # 2. Analyze active (newly trained) heads and layers
                active_modules = {k: v for k, v in model.named_children() if k != 'encoder'}
                dummy_active_model = torch.nn.ModuleDict(active_modules)
                watcher_act = ww.WeightWatcher(model=dummy_active_model)
                details_act = watcher_act.analyze()
                if details_act is not None and not details_act.empty and 'alpha' in details_act.columns:
                    act_alpha = details_act['alpha'].mean()
                    epoch_metrics["ww_alpha_active"] = act_alpha
                else:
                    act_alpha = 0.0
                    
                print(f"  [WeightWatcher] Encoder Alpha: {enc_alpha:.3f} | Active Heads Alpha: {act_alpha:.3f}")
            except ImportError:
                print("  [WeightWatcher] Not installed. Run `pip install weightwatcher` to see layer health metrics.")
            except Exception as e:
                print(f"  [WeightWatcher] Analysis skipped: {e}")

        # --- Offline Probes (Phase 1A only) ---
        log_loto_metrics = {}
        val_linear_acc = 0.0
        train_linear_acc = 0.0
        subj_cross_acc = 0.0
        if phase == '1a' and (epoch % probe_interval == 0 or epoch == epochs - 1):
            from test import _linear_probe_loto, _linear_probe_accuracy, _svm_probe_accuracy
            
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
            
            import copy
            clean_train_set = copy.copy(train_loader.dataset)
            clean_train_set.augmentations = None
            
            bs = getattr(train_loader, 'batch_size', None)
            if bs is None:
                try:
                    bs = clean_train_set.config.training.batch_size
                except Exception:
                    bs = 16
            
            clean_train_loader = DataLoader(clean_train_set, batch_size=bs, shuffle=False, num_workers=0)
            
            emb_t, lbl_t, grps_t = _extract_feats(clean_train_loader)
            emb_v, lbl_v, grps_v = _extract_feats(val_loader)
            
            subj_t = np.array([tr_b2s.get(str(g), -1) for g in grps_t])
            subj_v = np.array([vl_b2s.get(str(g), -1) for g in grps_v])
            
            suffix = os.path.basename(checkpoint_dir)
            num_classes = clean_train_set.config.model.get('num_classes', 5)
            
            loto_train_results = _linear_probe_loto(
                emb_t, lbl_t, grps_t, tag="Train", suffix=suffix, num_classes=num_classes
            )
            loto_val_results = _linear_probe_loto(
                emb_v, lbl_v, grps_v, tag="Val", suffix=suffix, num_classes=num_classes
            )
            
            train_linear_acc = loto_train_results.get("Eval/LOTO_Train_MeanAcc", 0.0)
            val_linear_acc = loto_val_results.get("Eval/LOTO_Val_MeanAcc", 0.0)
            lp_train_val = _linear_probe_accuracy(emb_t, lbl_t, emb_v, lbl_v)
            
            log_loto_metrics = {**loto_train_results, **loto_val_results}
            
            # Proxy-A
            subj_cross_acc = 0.0
            if not np.all(subj_t == -1):
                subj_cross_acc = _svm_probe_accuracy(emb_t, subj_t, emb_v, subj_v)
            
            print(f"  Offline Probes — Emotion Train LOTO Mean Acc: {train_linear_acc:.4f} | Val LOTO Mean Acc: {val_linear_acc:.4f} | Train->Val Acc: {lp_train_val:.4f}")
            if not np.all(subj_t == -1):
                print(f"  Proxy-A Probes — Subject Val Cross Acc (SVM): {subj_cross_acc:.4f}")

            # Live Tracker: probe results
            live_tracker.probe_results(phase, epoch, {
                "emotion_train_loto_mean": train_linear_acc,
                "emotion_val_loto_mean": val_linear_acc,
                "emotion_train_val_cross": lp_train_val,
                "subject_val_cross": subj_cross_acc,
            })

        log_dict = {
            f"Phase{phase}/Train/Loss": avg_train_loss,
            f"Phase{phase}/Val/Loss": val_loss,
            f"Phase{phase}/Val/Accuracy": val_acc,
            f"Phase{phase}/Val/Balanced_Accuracy": val_bal_acc,
            "LR": current_lr
        }

        # --- Checkpoint: checkpoint_last.pt (every epoch, atomic write) ---
        save_checkpoint(model, optimizer, scheduler, epoch, val_acc, 0.0,
                        f"{checkpoint_dir}/checkpoint_temp.pt",
                        momentum_model=momentum_model, supmoco_state=supmoco_state, smc_engine=smc_engine) 
        os.replace(f"{checkpoint_dir}/checkpoint_temp.pt", f"{checkpoint_dir}/checkpoint_last.pt")
        
        print(f"Phase {phase} Epoch {epoch} | LR: {current_lr:.2e} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")
        
        if use_supmoco and supmoco_state is not None:
            queue_len = int(supmoco_state.queue_filled.item())
            log_dict.update({
                f"Phase{phase}/Train/QueueSize": queue_len,
                f"Phase{phase}/Train/QueueFillRatio": queue_len / float(max(1, supmoco_state.queue_size)),
            })
        if smc_engine is not None:
            smc_engine.on_epoch_end()   # advances NullspaceState epoch (§11.4ii warmup gate)
            if last_smc_diag is None:
                raise RuntimeError("SMC engine ran an epoch with zero diagnostics — the train loop never "
                                   "reached the engine branch (check dual-view batch plumbing for loss.sm)")
            log_dict.update({
                f"Phase{phase}/SMC/Gate": last_smc_diag["gate"],
                f"Phase{phase}/SMC/LambdaOT": last_smc_diag["lambda_ot"],
                f"Phase{phase}/SMC/C_K": last_smc_diag["C_K"],
                f"Phase{phase}/SMC/AnnealT": last_smc_diag["T"],
                f"Phase{phase}/SMC/AlphaStd": last_smc_diag["alpha_std"],
                f"Phase{phase}/SMC/StdIntraTrial": last_smc_diag["stdt_intra"],
                f"Phase{phase}/SMC/EnergyRatio": last_smc_diag["energy_ratio"],
                f"Phase{phase}/SMC/QueueFill": last_smc_diag["queue"]["fill_ratio"],
                f"Phase{phase}/SMC/NullspaceRank": last_smc_diag["space"].get("rank", 0),
                f"Phase{phase}/SMC/ColdAlphaRows": last_smc_diag["cold_alpha_rows"],
            })
            for _y, _dv in last_smc_diag["D_norm"].items():
                log_dict[f"Phase{phase}/SMC/DNorm_y{_y}"] = float(_dv)
            # B3(iii) early stop: Stdt(alpha) < 0.02 for 3 consecutive epochs AFTER the
            # anneal floor means alpha has degenerated to the constant rho prior — the
            # gate carries no within-trial information. Stop loudly; ladder next.
            smc_std_series.append(float(last_smc_diag["stdt_intra"]))
            smc_anneal_series.append(bool(last_smc_diag["anneal_complete"]))
            _escape = alpha_identity_escape(smc_std_series, smc_anneal_series)
            if _escape is not None:
                raise RuntimeError(
                    f"B3(iii) EARLY STOP at epoch {epoch}: intra-trial Stdt(alpha) < 0.02 "
                    "for 3 consecutive post-anneal epochs — alpha == rho (uninformative "
                    "gate). Run certification.remediation_ladder on the offending class in "
                    "STRICT causal order (eps position -> rho seed/s -> N_q/ESS/anneal T); "
                    "see CERTIFICATION_RUNBOOK.md.")
        if phase == '1a' and (epoch % probe_interval == 0 or epoch == epochs - 1):
            log_dict[f"Phase{phase}/Train/Emotion_LOTO_Mean_Acc"] = train_linear_acc
            log_dict[f"Phase{phase}/Val/Emotion_LOTO_Mean_Acc"] = val_linear_acc
            log_dict[f"Phase{phase}/Val/Emotion_Probe_Val_Acc"] = lp_train_val
            log_dict[f"Phase{phase}/Val/Subject_Probe_Val_Acc"] = subj_cross_acc
        log_dict.update(log_loto_metrics)

        for k, v in log_dict.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                writer.add_scalar(k, v, epoch)

        # --- Phase 1A: Two-tier checkpointing (best_metric makes the criterion explicit) ---
        # checkpoint_best.pt: save when train loss decreases
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            save_checkpoint(model, optimizer, scheduler, epoch, val_acc, 0.0,
                            f"{checkpoint_dir}/checkpoint_temp.pt",
                            momentum_model=momentum_model, supmoco_state=supmoco_state, smc_engine=smc_engine,
                            best_metric={'name': 'train_loss', 'value': avg_train_loss})
            os.replace(f"{checkpoint_dir}/checkpoint_temp.pt", f"{checkpoint_dir}/checkpoint_best.pt")
            wandb.save(f"{checkpoint_dir}/checkpoint_best.pt")
            live_tracker.checkpoint_saved(f"{checkpoint_dir}/checkpoint_best.pt",
                                         "train_loss", avg_train_loss)
        else:
            print(f"Best Train Loss yet: {best_train_loss:.4f}")

        # checkpoint_best_linear.pt: save when offline linear eval accuracy increases
        if val_linear_acc > best_linear_acc:
            best_linear_acc = val_linear_acc
            save_checkpoint(model, optimizer, scheduler, epoch, val_acc, 0.0,
                            f"{checkpoint_dir}/checkpoint_temp.pt",
                            momentum_model=momentum_model, supmoco_state=supmoco_state, smc_engine=smc_engine,
                            best_metric={'name': 'linear_eval_acc', 'value': val_linear_acc})
            os.replace(f"{checkpoint_dir}/checkpoint_temp.pt", f"{checkpoint_dir}/checkpoint_best_linear.pt")
            wandb.save(f"{checkpoint_dir}/checkpoint_best_linear.pt")
            print(f"  New best Linear Eval Acc: {best_linear_acc:.4f}")
            live_tracker.checkpoint_saved(f"{checkpoint_dir}/checkpoint_best_linear.pt",
                                         "linear_eval_acc", best_linear_acc)

        # Inject all dynamic custom metrics from epoch_metrics into log_dict for W&B
        log_dict[f"Phase{phase}/Train/Emotion_Accuracy"] = epoch_metrics.get("train_acc", 0.0)
        
        if "ww_alpha_encoder" in epoch_metrics:
            log_dict[f"Phase{phase}/Metrics/WeightWatcher_Encoder_Alpha"] = epoch_metrics["ww_alpha_encoder"]
        if "ww_alpha_active" in epoch_metrics:
            log_dict[f"Phase{phase}/Metrics/WeightWatcher_Active_Alpha"] = epoch_metrics["ww_alpha_active"]

        wandb.log(log_dict, step=epoch)

        # Live Tracker: epoch summary (after all metrics computed)
        live_tracker.epoch_summary(phase, epoch, epochs, epoch_metrics)


def phase_one_a(config, model, device, train_loader, val_loader, writer, checkpoint_dir, resume, checkpoint=None):
    print("=== Phase 1A: Training Encoder Only (Old Style, No MIL/Bagging, No SwiGLU) ===")

    # Live tracker (Telegram/Discord)
    live_tracker = LiveTracker(config=config)
    
    if config.training.get('bagging', False):
        print("WARNING: 'bagging' is set to True in config, but Phase 1A enforces old-style 1s windows. Ensure dataset complies!")

    if isinstance(model, SpikingMobileNetProjector):
        enc_class = model
    else:
        enc_class = SpikingMobileNetProjector(
            encoder_backbone = model.encoder,
            num_classes=config.model.get('num_classes', 5)
        ).to(device)

    use_scda = config.loss.get('use_scda', False)
    iic_intra_weight = config.loss.get('iic_intra_weight', 1.0)
    iic_inter_weight = config.loss.get('iic_inter_weight', 1.0)
    decoupled = config.loss.get('decoupled', False)
    con_temp = config.loss.get('temperature', 0.07)
    use_supmoco = config.training.get('use_supmoco', False)
    _sm_cfg = config.loss.get('sm', None) if hasattr(config, 'loss') else None
    use_sm = _sm_cfg is not None and bool(_sm_cfg.get('enabled', False))
    if use_sm:
        # SMC mode reuses the dual-view batch + momentum-encoder plumbing; loss_fn below
        # is the VALIDATION criterion only (training loss = SMCLossEngine, built with the
        # state). The legacy SupMoCoLoss/SupMoCoState objects are never constructed.
        use_supmoco = True
        loss_fn = nn.CrossEntropyLoss()
    elif use_supmoco:
        loss_fn = SupMoCoLoss(
            train_loader.dataset.prototypes,
            temperature=con_temp,
            iic_enabled=use_scda,
            iic_intra_weight=iic_intra_weight,
            iic_inter_weight=iic_inter_weight,
            temporal_decay_enabled=config.training.get('temporal_decay_enabled', False),
            temporal_decay_factor=config.training.get('temporal_decay_factor', 0.999),
            exclude_same_trial=config.loss.get('exclude_same_trial', True),
            cosface_margin=config.loss.get('cosface_margin', 0.0)
        )
    else:
        loss_fn = ContrastiveLoss(
            train_loader.dataset.prototypes,
            temperature=con_temp,
            iic_enabled=use_scda,
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
    best_train_loss = float('inf')
    if resume and checkpoint is not None:
        enc_class.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['accuracy']
        best_dice = checkpoint['dice']
        _bm = checkpoint.get('best_metric')
        if _bm and _bm.get('name') == 'train_loss':
            best_train_loss = float(_bm.get('value', float('inf')))
        print(f"Resuming training from epoch {start_epoch}... (best train loss so far: {best_train_loss:.4f})")

    # Build momentum encoder AFTER loading weights so it inherits restored state
    momentum_model = None
    supmoco_state = None
    if use_supmoco:
        momentum_model = build_momentum_encoder(enc_class).to(device)
        if not use_sm:
            supmoco_state = SupMoCoState(
                queue_size=config.training.get('supmoco_queue_size', 4096),
                feature_dim=config.training.get('supmoco_feature_dim', 128),
            ).to(device)
        # Restore saved momentum/queue state if available
        if resume and checkpoint is not None:
            if checkpoint.get('momentum_model_state_dict') is not None:
                momentum_model.load_state_dict(checkpoint['momentum_model_state_dict'])
                print("  Restored momentum encoder weights from checkpoint.")
            if supmoco_state is not None and checkpoint.get('supmoco_state_dict') is not None:
                supmoco_state.load_state_dict(checkpoint['supmoco_state_dict'], strict=True)
                print(f"  Restored SupMoCo queue (filled={int(supmoco_state.queue_filled.item())}).")

    # ---- SMC engine (loss.sm): Eq. 23 assembled objective (plan §5 Stage 6) ----
    smc_engine = None
    if use_sm:
        num_classes_sm = int(config.model.get('num_classes', 5))
        smc_engine = SMCLossEngine(
            _sm_cfg,
            feature_dim=int(_sm_cfg.get('feature_dim', 256)),
            num_classes=num_classes_sm,
            num_subjects=int(_sm_cfg.get('num_subjects', 15)),
            total_steps=max(1, int(epochs) * max(1, len(train_loader))),
        ).to(device)
        smc_engine.space.set_epoch(start_epoch)   # resume keeps the §11.4ii warmup gate honest
        if resume and checkpoint is not None:
            if checkpoint.get('sm_state_dict') is None:
                raise RuntimeError("Resuming a loss.sm run from a pre-SMC checkpoint — it carries no "
                                   "sm_state_dict; start a fresh 1A run or resume an SMC checkpoint "
                                   "(no silent cold-start resume of rho/queue/geometry)")
            smc_engine.load_state_dict(checkpoint['sm_state_dict'], strict=True)
            smc_engine.load_sm_state(checkpoint['sm_state'])
            print(f"  Restored SMC engine (step={smc_engine.sm_state()['step']}, eps={smc_engine.eps:.4f}).")
        else:
            # §9.5 Option-B pre-pass: standardized RAW WINDOW ||r||² per emotion class
            norms_by_class, scanned = {}, 0
            cap = int(_sm_cfg.get('seed_max_batches', 64))
            need = int(_sm_cfg.get('seed_min_windows', 512))
            with torch.no_grad():
                for _b in train_loader:
                    _view, _lab = (_b[0], _b[3]) if len(_b) == 8 else (_b[0], _b[2])
                    n2 = (_view.to(torch.float64).reshape(_view.shape[0], -1) ** 2).sum(dim=1).cpu()
                    _lab = _lab.view(-1)
                    for y in _lab.unique().tolist():
                        norms_by_class.setdefault(int(y), []).append(n2[_lab == y])
                    scanned += 1
                    if scanned >= cap and all(sum(t.numel() for t in norms_by_class.get(y, [])) >= need
                                              for y in range(num_classes_sm)):
                        break
            starved = [y for y in range(num_classes_sm)
                       if sum(t.numel() for t in norms_by_class.get(y, [])) < need]
            if starved:
                raise RuntimeError(f"Option-B seed pre-pass has <{need} windows for classes {starved} after "
                                   f"{scanned} batches — raise loss.sm.seed_max_batches/seed_min_windows or "
                                   "fix the split (a starved Option-B seed is refused, not warned)")
            seed_info = smc_engine.seed_priors({y: torch.cat(norms_by_class[y]) for y in range(num_classes_sm)})
            print("  [SMC] Option-B priors: "
                  + ", ".join(f"rho_{y}={v:.4f}" for y, v in seed_info['rho'].items())
                  + f" | eps={seed_info['eps']:.4f}")
        smc_engine.run_sizing_checks(
            batch_size=int(config.training.get('batch_size', 64)),
            epochs=int(epochs), steps_per_epoch=max(1, len(train_loader)),
            L_const=float(_sm_cfg.get('sizing_L', 1.0)),
            d_eff=float(_sm_cfg.get('sizing_d_eff', _sm_cfg.get('feature_dim', 256))),
            tol_eff=float(_sm_cfg.get('sizing_tol', 0.1)),
            C_const=float(_sm_cfg.get('sizing_C_const', 4.0)),
            eps_tol=float(_sm_cfg.get('sizing_eps_tol', 1e-2)))
        print("  [SMC] §9.6 sizing + §9.4 staleness init gates passed")



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
        smc_engine=smc_engine,
        unfreeze_epoch=-1,
        accumulation_steps=accumulation_steps,
        best_train_loss=best_train_loss,
        live_tracker=live_tracker
    )

    if smc_engine is not None:
        lock = smc_engine.lock_fullpass()
        lock_path = f"{checkpoint_dir}/lockpass_1a.pt"
        torch.save(lock, lock_path)
        print(f"  [SMC] lockpass_1a artifact written: {lock_path} "
              f"(rank={int(lock['space'].get('rank', -1))}, clouds={sorted(lock['clouds'])})")
   

def phase_two(config, model, device, train_loader, val_loader, writer, checkpoint_dir, resume, checkpoint):
    raise RuntimeError(
        "Phase 2 was retired in Stage 0 tranche 2 of the SMC rebuild (seg/UNet path "
        "deleted; see REVIEW_SMC_AUDIT.md §5/§8). SNN stages return after Stage 7.")


def phase_three(config, model, device, train_loader, val_loader, writer, checkpoint_dir, resume, checkpoint):
    raise RuntimeError(
        "Phase 3 was retired in Stage 0 tranche 2 of the SMC rebuild (see REVIEW_SMC_AUDIT.md §5/§8).")


phase_handles = {
    '1a': phase_one_a,
    '1': phase_one_a,  # fallback for configs specifying '1'
    '1b': None,  # retired in Stage 0 tranche 2; rebuilt as the R_b SMC adapter at plan Stage 7
    '2': None,
    '3': None,
}


def run_training(config, model, device, phase, resume, loso=None, subj=None, checkpoint=None):

    if phase not in phase_handles:
        raise ValueError(f"Unknown phase {phase!r}; supported: {sorted(phase_handles)}")
    if phase_handles[phase] is None:
        raise RuntimeError(
            f"Phase {phase!r} was retired in Stage 0 tranche 2 of the SMC rebuild "
            "(legacy MIL/VIB/DANN-1B and the seg-UNet 2/3 paths deleted; "
            "see REVIEW_SMC_AUDIT.md §5/§8). Phase 1B returns at plan Stage 7.")

    logger_cfg = config.get('logger', {})
    logging_enabled = bool(logger_cfg.get('enabled', True))
    if not logging_enabled:
        print("Logger disabled via config.logger.enabled = False — "
              "continuing training with W&B disabled (B9).")

    # Use pre-calculated run_name from main.py if available, otherwise generate one
    run_name = logger_cfg.get('run_name')
    if not run_name:
        time_str = datetime.now().strftime('%m%d_%H%M%S')
        id_tag = f"loso{loso}" if loso else (f"subj{subj}" if subj else "run")
        run_name = f"{logger_cfg.get('run_name', config.experiment_name)}_{time_str}_{id_tag}"
        
        if 'logger' in config:
            from omegaconf import open_dict
            with open_dict(config):
                config.logger.run_name = run_name

    checkpoint_dir = os.path.join(config.data.get('save_path', ''), "saved_models", f"phase{phase}", run_name)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if wandb.run is None:
        wandb.init(
            project=logger_cfg.get('project_name', 'snn'),
            name=run_name,
            config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
            tags=list(logger_cfg.get('tags', [])),
            mode="disabled" if (logger_cfg.get('offline') or not logging_enabled) else "online",
            settings=wandb.Settings(_disable_stats=True, _disable_meta=True) 
        )
    else:
        print(f"[W&B] Using existing run: {wandb.run.name}")
    
    log_dir = os.path.join("results", run_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"Initializing TensorBoard: {log_dir}")

    masks = torch.load(os.path.join(config.data.dataset_path,'masks.pt')).to(device)
    density = masks.sum() / masks.numel()
    print(f"Global Density (Consider this for setting up target firing rate): {density:.4f}")

    # Create dataset first (without augmentations) so FourierMixup can scan it
    train_set = SWEEPDataset(
        config,
        split='train',
        loso=loso,
        subj=subj,
        prototypes=masks,
        augmentations=None,  # assigned below after FourierMixup scan
    )

    # --- Fourier Mixup: data-level identity erasure (replaces DANN/MMD) ---
    use_fourier_mixup = config.training.get('use_fourier_mixup', False)
    fourier_mixup_modules = []
    if use_fourier_mixup:
        fmix_cfg = config.training.get('fourier_mixup', {})
        fmix_sigma = fmix_cfg.get('sigma', None)
        fmix_retention = fmix_cfg.get('retention_ratio', 0.95)
        fmix_p = fmix_cfg.get('p', 0.5)
        fmix_beta = fmix_cfg.get('beta_alpha', 1.0)
        fmix_grid = config.data.get('grid_size', 32)
        
        # Always use build_fourier_mixup (handles caching + donor pool)
        fourier_mixup_module = build_fourier_mixup(
            train_set, retention_ratio=fmix_retention, p=fmix_p, 
            beta_alpha=fmix_beta, device=str(device),
            cache_dir=config.data.dataset_path, loso_id=loso
        )
        # Override sigma if explicitly specified
        if fmix_sigma is not None:
            print(f"[FourierMixup] Overriding \u03c3 with pre-specified value: {fmix_sigma}")
            fourier_mixup_module.sigma = fmix_sigma
            # Recompute the Gaussian mask with the new sigma
            fmix_grid = config.data.get('grid_size', 32)
            cy, cx = fmix_grid // 2, fmix_grid // 2
            yy, xx = torch.meshgrid(torch.arange(fmix_grid, dtype=torch.float32),
                                     torch.arange(fmix_grid, dtype=torch.float32), indexing='ij')
            dist_sq = (yy - cy) ** 2 + (xx - cx) ** 2
            fourier_mixup_module.gaussian_mask = torch.exp(-dist_sq / (2.0 * fmix_sigma ** 2))
        fourier_mixup_modules = [fourier_mixup_module]
    
    # Compose: FourierMixup (if enabled) → standard augmentations
    all_aug_modules = fourier_mixup_modules + [
        GaussianNoise(std=0.05),
        FrequencyDropout(p=0.1),
        VideoRandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        VideoTemporalMasking(p=0.1, max_mask_len=8),
        SignalJitter(lower=0.5, upper=2.0)
    ]
    train_aug = nn.Sequential(*all_aug_modules)
    train_set.augmentations = train_aug  # Now assign augmentations
    
    val_set = SWEEPDataset(
        config,
        split='val',
        loso=loso,
        subj=subj,
        prototypes=masks,
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