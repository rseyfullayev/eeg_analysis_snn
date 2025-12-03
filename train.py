import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.snn_modeling.utils.loss import FullHybridLoss, FiringRateRegularizer, TopKClassificationLoss
from src.snn_modeling.dataloader.dataset import SWEEPDataset
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import snntorch as snn
import segmentation_models_pytorch as smp
import wandb
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import numpy as np
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from src.snn_modeling.utils.utils import initialize_network, monitor_network_health
from src.snn_modeling.layers.neurons import ALIF

# Ignore the specific sklearn warning about missing classes
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
warnings.filterwarnings("ignore", message="A single label was found in 'y_true' and 'y_pred'. For the confusion matrix to have the correct shape, use the 'labels' parameter to pass all known labels.")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)



def save_checkpoint(model, optimizer, scheduler, epoch, acc, dice, path="best_sweepnet.pt"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy': acc,
        'dice': dice
    }, path)
    print(f"New Record! Saved checkpoint at Epoch {epoch} (Acc: {acc:.4f}, Dice: {dice:.4f})")

def validate(model, val_loader, criterion, device, threshold=0.5):

    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []

    tp_tot, fp_tot, fn_tot, tn_tot = 0, 0, 0, 0

    with torch.no_grad():
        for inputs, targets, labels in val_loader:
            inputs, targets, labels = inputs.to(device), targets.to(device), labels.to(device)

            inputs = inputs.permute(1, 0, 2, 3, 4)
            
            
            outputs = model(inputs)
            outputs_avg = torch.mean(outputs, dim=0)
            
            loss = criterion(outputs_avg, targets, labels)
            val_loss += loss.item()
            B, C, H, W = outputs_avg.shape
            k_percent = loss.k_percent if hasattr(loss, 'k_percent') else 0.1
            k = max(1, int(H * W * k_percent))
            flat_logits = outputs_avg.view(B, C, -1)
            top_k_values, _ = torch.topk(flat_logits, k, dim=2)
            energy_logits = torch.mean(top_k_values, dim=2)
            preds = torch.argmax(energy_logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            soft_probs = torch.sigmoid(outputs_avg)
  
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

            target_masks = (targets > threshold).long()
            

            tp, fp, fn, tn = smp.metrics.get_stats(
                soft_probs, target_masks, mode='multilabel', threshold=threshold
            )
            tp_tot += tp.sum().item()
            fp_tot += fp.sum().item()
            fn_tot += fn.sum().item()
            tn_tot += tn.sum().item()

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
        inputs, targets, labels = next(data_iter)
    except StopIteration:
        return 

    inputs, targets, labels = inputs.to(device), targets.to(device), labels.to(device)
    inputs_snn = inputs.permute(1, 0, 2, 3, 4)

    with torch.no_grad():
        outputs = model(inputs_snn)
        outputs_avg = torch.mean(outputs, dim=0)
        probs = torch.sigmoid(outputs_avg)

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
        
        writer.add_image(f"Vis/Input_{idx}", img_input[None, ...], epoch)
        writer.add_image(f"Vis/Target_{idx}", img_target[None, ...], epoch)
        writer.add_image(f"Vis/Pred_{idx}", img_pred[None, ...], epoch)
        
        wandb.log({
            f"Visuals/Sample_{idx}": [
                wandb.Image(img_input, caption=f"Input (Avg Activity) - {caption_text}"),
                wandb.Image(img_target, caption=f"Target {caption_text})"),
                wandb.Image(img_pred, caption=f"Prediction - {caption_text}"),
                wandb.Image(img_pred_bin, caption=f"Hard Pred (Thresh {threshold})")
            ]
        }, step=epoch)

def create_optimizer(model, config):
    lr = config['training'].get('learning_rate', 1e-3)
    base_params = []
    base_params_no_decay = []
    time_params = []
    threshold_params = []
    no_decay_id = set()
    classess_names = (snn.Leaky, snn.Synaptic, snn.Alpha, 
                      ALIF, TopKClassificationLoss, 
                      nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, 
                      nn.LayerNorm, nn.GroupNorm)
    for m in model.modules():
        if isinstance(m, classess_names):
            for param in m.parameters(recurse=False):
                no_decay_id.add(id(param))    
        if hasattr(m, 'bias') and m.bias is not None:
            no_decay_id.add(id(m.bias))
        if hasattr(m, 'gain') and m.gain is not None:
             no_decay_id.add(id(m.gain))

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'alpha' in name or 'beta' in name or 'slope' in name:
            time_params.append(param)
        elif 'threshold' in name:
            threshold_params.append(param)
        elif id(param) in no_decay_id:
            base_params_no_decay.append(param)
        else:
            base_params.append(param)

    optimizer = optim.AdamW([
        {'params': base_params, 'lr': lr, 'weight_decay': config['training'].get('weight_decay', 1e-4)},
        {'params': base_params_no_decay, 'lr': lr, 'weight_decay': 0.0},
        {'params': time_params, 'lr': lr * 1e-2, 'weight_decay': 0.0},
        {'params': threshold_params, 'lr': lr * 100, 'weight_decay': 0.0}
    ], betas=(0.9, 0.999))

    return optimizer

def run_training(config, model, device, checkpoint=None):

    run_name = f"{config['experiment_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = os.path.join("saved_models", run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    wandb.init(
        project=config['logging']['project_name'],
        name=config['logging']['run_name'],
        config=config,
        tags=config['logging']['tags'],
        mode="disabled" if config['logging'].get('offline') else "online",

        settings=wandb.Settings(_disable_stats=True, _disable_meta=True) 
    )
    
    log_dir = os.path.join("results", run_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"Initializing TensorBoard: {log_dir}")

    data_path = os.path.join(config['data']['dataset_path'], "data.npy")
    label_path = os.path.join(config['data']['dataset_path'], "labels.npy")
    
    # Load as Tensor
    all_data = torch.tensor(np.load(data_path), dtype=torch.float32)
    all_labels = torch.tensor(np.load(label_path), dtype=torch.long)
    num_samples = len(all_labels)
    indices = np.arange(num_samples)

    train_idx, val_idx = train_test_split(
        indices, 
        test_size=(1 - config['data']['train_split']), 
        random_state=42, 
        stratify=all_labels # Good for imbalanced emotions!
    )

    train_data_slice = all_data[train_idx]
    train_label_slice = all_labels[train_idx]
    '''
    master_prototypes = SWEEPDataset.compute_prototypes(
        train_data_slice, 
        train_label_slice, 
        config['model']['num_classes'],
        config['data']['grid_size']
    )
    '''
    prototype_bank = SWEEPDataset.compute_prototypes(None, None, 
                                                    config['model']['num_classes'],
                                                    config['data']['grid_size'], device=device)
    train_set = SWEEPDataset(
        config, 
        data=train_data_slice, 
        labels=train_label_slice, 
        prototypes=prototype_bank, #master_prototypes, 
        mode='manual'
    )
    
    val_set = SWEEPDataset(
        config, 
        data=all_data[val_idx], 
        labels=all_labels[val_idx], 
        prototypes=prototype_bank, #master_prototypes,
        mode='manual'
    )
    
    train_loader = DataLoader(train_set, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['data'].get('num_workers', 0))
    val_loader = DataLoader(val_set, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['data'].get('num_workers', 0))
    print(f"Data Loaded: {len(train_set)} Train | {len(val_set)} Val")

    loss_fn = FullHybridLoss(
        smooth = config['loss'].get('smooth', 0.0),
        lambda_seg = config['loss'].get('lambda_seg', 1.0),
        lambda_bce = config['loss'].get('lambda_bce', 1.0),
        lambda_class = config['loss'].get('lambda_class', 1.0),
        alpha = config['loss'].get('alpha', 0.5),
        beta = config['loss'].get('beta', 0.5),
        label_smoothing = config['loss'].get('label_smoothing', 0.0)

    )
    
    spike_tax_collector = FiringRateRegularizer(model, 
                                                target_rate=config['loss'].get('target_rate', 0.05), 
                                                lambda_reg=config['loss'].get('lambda_fire', 0.1))
    

    optimizer = create_optimizer(model, config)
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'], eta_min=1e-6)
    
    start_epoch = 0
    best_acc = 0.0
    best_dice = 0.0
    epochs = config['training']['epochs']

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['accuracy']
        best_dice = checkpoint['dice']
        print(f"Resuming training from epoch {start_epoch}...")

    model = model.to(device)
    
    initialize_network(model, train_loader, device)

    for epoch in range(start_epoch, epochs):
        
        model.train()
        train_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}", unit="batch")
        for _, (inputs, targets, targets_c) in enumerate(train_loop):

            inputs, targets, targets_c = inputs.to(device), targets.to(device), targets_c.to(device)
            inputs = inputs.permute(1, 0, 2, 3, 4)

            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                outputs = model(inputs)
                outputs_agg = torch.mean(outputs, dim=0)
                loss_spike_tax = spike_tax_collector.compute_tax()
                loss = loss_fn(outputs_agg, targets, targets_c) + loss_spike_tax
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        val_loss, val_acc, val_bal_acc, val_dice, val_iou, val_pre, val_rec = validate(model, val_loader, loss_fn, device)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch} | LR: {current_lr:.2e} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | Dice: {val_dice:.4f} | "
              f"Val Pre: {val_pre:.4f} | Val Rec: {val_rec:.4f}")
        
        log_dict = {
            "Train/Loss": avg_train_loss,
            "Val/Loss": val_loss,
            "Val/Accuracy": val_acc,
            "Val/Balanced_Accuracy": val_bal_acc,
            "Val/Dice": val_dice,
            "Val/IoU": val_iou,
            "Val/Precision": val_pre,
            "Val/Recall": val_rec,
            "LR": current_lr
        }
        wandb.log(log_dict, step=epoch)
        for k, v in log_dict.items(): writer.add_scalar(k, v, epoch)
 
        if (epoch + 1) % config['logging'].get('log_interval', 10) == 0:
            print(f"Logging Visuals for Epoch {epoch}...")
            log_visuals(model, val_loader, device, writer, epoch) 
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_dice = val_dice
            save_checkpoint(model, optimizer, scheduler, epoch, best_acc, best_dice, f"{checkpoint_dir}/checkpoint_{epoch:03d}_{best_acc:.4f}_{best_dice:.4f}.pt")
        elif val_acc == best_acc and val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(model, optimizer, scheduler, epoch, best_acc, best_dice, f"{checkpoint_dir}/checkpoint_{epoch:03d}_{best_acc:.4f}_{best_dice:.4f}.pt")
            
    
    print("--- Training Complete ---")
    wandb.finish()
    writer.close()