import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.snn_modeling.utils.loss import FullHybridLoss, TopKClassificationLoss
from src.snn_modeling.dataloader.dataset import SWEEPDataset
import time
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

from tqdm import tqdm
from src.snn_modeling.utils.utils import initialize_network
from src.snn_modeling.layers.neurons import ALIF
from src.snn_modeling.models.unet import SpikingResNetClassifier
from src.snn_modeling.models.encoders import SpikingResNet18Encoder

# Ignore the specific sklearn warning about missing classes
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
warnings.filterwarnings("ignore", message="A single label was found in 'y_true' and 'y_pred'. For the confusion matrix to have the correct shape, use the 'labels' parameter to pass all known labels.")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

#torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True
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

def validate(model, val_loader, criterion, device, threshold=0.5, only_classification=False):

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
            B,C,T,H,W = inputs.shape
            flat = inputs.view(B, -1).abs()
            p98 = torch.quantile(flat, 0.98, dim=1, keepdim=True).view(B, 1, 1, 1, 1)
            inputs = torch.tanh(inputs / (p98 + 1e-6) * 3.0)
            inputs = inputs.permute(2, 0, 1, 3, 4)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets, labels)

            val_loss += loss.item()

            B, C, H, W = outputs.shape
            k_percent = loss.k_percent if hasattr(loss, 'k_percent') else 0.1
            k = max(1, int(H * W * k_percent))
            flat_logits = outputs.view(B, C, -1)
            top_k_values, _ = torch.topk(flat_logits, k, dim=2)
            energy_logits = torch.mean(top_k_values, dim=2)
            preds = torch.argmax(energy_logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
            if not only_classification:
                soft_probs = torch.sigmoid(outputs)
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

def freeze_bn_stats(module):
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
            module.eval()

def create_optimizer(model, loss_fn, config, low_encoder_lr=False):
    lr = config['training'].get('learning_rate', 1e-3)
    encoder_params = []
    base_params = []
    base_params_no_decay = []
    time_params = []
    threshold_params = []
    no_decay_id = set()
    classess_names = (snn.Leaky, snn.Synaptic, snn.Alpha, 
                      ALIF, TopKClassificationLoss, 
                      nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, 
                      nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)
    for m in model.modules():
        if isinstance(m, classess_names):
            for param in m.parameters(recurse=False):
                no_decay_id.add(id(param))    
        if hasattr(m, 'bias') and m.bias is not None:
            no_decay_id.add(id(m.bias))
        if hasattr(m, 'gain') and m.gain is not None:
             no_decay_id.add(id(m.gain))
    for m in loss_fn.modules():
        if isinstance(m, classess_names):
            for param in m.parameters(recurse=False):
                no_decay_id.add(id(param))      

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'alpha' in name or 'beta' in name or 'slope' in name or 'decay' in name or 'gamma' in name:
            time_params.append(param)
            print(f"Special LR for Time Param: {name}")
        elif 'threshold' in name:
            threshold_params.append(param)
            print(f"Special LR for Threshold Param: {name}")
        elif id(param) in no_decay_id:
            base_params_no_decay.append(param)
            print(f"No Decay Param: {name}")
        elif low_encoder_lr and 'encoder' in name:
            encoder_params.append(param)
            print(f"Encoder Low LR Param: {name}")
        else:
            base_params.append(param)
            print(f"Base Decay Param: {name}")

    for name, param in loss_fn.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in no_decay_id:
            base_params_no_decay.append(param)
            print(f"No Decay Param: {name}")
        else:
            base_params.append(param)
            print(f"Base Decay Param: {name}")
    optimizer = optim.AdamW([
        {'params': base_params, 'lr': lr, 'weight_decay': config['training'].get('weight_decay', 1e-4)},
        {'params': encoder_params, 'lr': lr * 1e-2, 'weight_decay': config['training'].get('weight_decay', 1e-4)},
        {'params': base_params_no_decay, 'lr': lr * 1e-1, 'weight_decay': 0.0},
        {'params': time_params, 'lr': lr * 1e-2, 'weight_decay': 0.0},
        {'params': threshold_params, 'lr': lr * 100, 'weight_decay': 0.0}
    ], betas=(0.9, 0.999))

    return optimizer


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
                  freeze_bn=False):


    for epoch in range(start_epoch, epochs):
        model.train()

        if freeze_bn:
            model.encoder.apply(freeze_bn_stats)
        train_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Phase {phase} Epoch {epoch+1}/{epochs}", unit="batch")
        for batch_idx, (inputs, targets, targets_c) in enumerate(train_loop):
            
            inputs, targets, targets_c = inputs.to(device), targets.to(device), targets_c.to(device)
            B,C,T,H,W = inputs.shape
            flat = inputs.view(B, -1).abs()
            p98 = torch.quantile(flat, 0.98, dim=1, keepdim=True).view(B, 1, 1, 1, 1)
            inputs = torch.tanh(inputs / (p98 + 1e-6) * 3.0)
            inputs = inputs.permute(2, 0, 1, 3, 4)
            outputs = model(inputs)

            loss = loss_fn(outputs, targets, targets_c)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
          
        val_loss, val_acc, val_bal_acc, val_dice, val_iou, val_pre, val_rec = validate(model, val_loader, loss_fn, device, only_classification=(phase == 1))
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        if phase == 1:
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
        if phase != 1:
            log_dict.update({
                f"Phase{phase}/Val/Dice": val_dice,
                f"Phase{phase}/Val/IoU": val_iou,
                f"Phase{phase}/Val/Precision": val_pre,
                f"Phase{phase}/Val/Recall": val_rec,
            })

        wandb.log(log_dict, step=epoch)
        for k, v in log_dict.items(): writer.add_scalar(k, v, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            best_dice = val_dice
            save_checkpoint(model, optimizer, scheduler, epoch, best_acc, best_dice, f"{checkpoint_dir}/checkpoint_{epoch:03d}_{best_acc:.4f}_{best_dice:.4f}.pt")
        
        elif val_acc == best_acc and val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(model, optimizer, scheduler, epoch, best_acc, best_dice, f"{checkpoint_dir}/checkpoint_{epoch:03d}_{best_acc:.4f}_{best_dice:.4f}.pt")


def phase_one(config, model, device, train_loader, val_loader, writer, checkpoint_dir, resume, checkpoint=None):
    print("=== Phase One: Training Encoder Only ===")
    enc_class = SpikingResNetClassifier(
        encoder_backbone = model.encoder,
        num_classes=config['model'].get('num_classes', 5)
    ).to(device)

    initialize_network(enc_class, train_loader, device)
    loss_fn = FullHybridLoss(
        smooth = 0.,
        lambda_seg = config['loss'].get('lambda_seg', 1.0),
        lambda_con = config['loss'].get('lambda_con', 0.0),
        lambda_class = config['loss'].get('lambda_class', 1.0),
        alpha = 0.,
        beta = 0.,
        time_steps=config['data'].get('num_timesteps', 16),
    )

    loss_fn.to(device)

    start_epoch = 0
    best_acc = 0.0
    best_dice = 0.0
    epochs = config['training']['phase_1_epochs']
    accumulation_steps = config['training'].get('accumulation_steps', 1)
    optimizer = create_optimizer(enc_class, loss_fn, config, low_encoder_lr=False)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    if resume:
        enc_class.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['accuracy']
        best_dice = checkpoint['dice']
        print(f"Resuming training from epoch {start_epoch}...")

    training_loop(1, start_epoch, epochs, best_acc, best_dice, enc_class, device, train_loader, val_loader, loss_fn, optimizer, scheduler, writer, checkpoint_dir)
    
    
def phase_two(config, model, device, train_loader, val_loader, writer, checkpoint_dir, resume, checkpoint):
    print("=== Phase Two: Training Rest Only ===")
    initialize_network(model, train_loader, device)
    loss_fn = FullHybridLoss(
        smooth = 0.0,
        lambda_seg = config['loss'].get('lambda_seg', 1.0),
        lambda_con = 0.0,
        lambda_class = config['loss'].get('lambda_class', 1.0),
        alpha = config['loss'].get('alpha', 0.5),
        beta = config['loss'].get('beta', 0.5),
        time_steps=config['data'].get('num_timesteps', 16),
    )

    loss_fn.to(device)
    if not resume:
        new_dict = {k:v for k,v in checkpoint['model_state_dict'].items() if 'encoder' in k}
        model.load_state_dict(new_dict, strict=False)

    start_epoch = 0
    best_acc = 0.0
    best_dice = 0.0
    epochs = config['training']['phase_2_epochs']
    accumulation_steps = config['training'].get('accumulation_steps', 1)

    for param in model.encoder.parameters():
        param.requires_grad = False

    optimizer = create_optimizer(model, loss_fn, config, low_encoder_lr=False)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    

    if resume:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['accuracy']
        best_dice = checkpoint['dice']
        print(f"Resuming training from epoch {start_epoch}...")
    
    training_loop(2, start_epoch, epochs, best_acc, best_dice, model, device, train_loader, val_loader, loss_fn, optimizer, scheduler, writer, checkpoint_dir)


def phase_three(config, model, device, train_loader, val_loader, writer, checkpoint_dir, resume, checkpoint):  
    print("=== Phase Three: Training Model ===")
    
    loss_fn = FullHybridLoss(
        smooth = 0.0,
        lambda_seg = config['loss'].get('lambda_seg', 1.0),
        lambda_con = 0.0,
        lambda_class = config['loss'].get('lambda_class', 1.0),
        alpha = config['loss'].get('alpha', 0.5),
        beta = config['loss'].get('beta', 0.5),
        time_steps=config['data'].get('num_timesteps', 16),
    )

    loss_fn.add_fire_rate_loss(model, 
                               lambda_fire=config['loss'].get('lambda_fire', 0.1), 
                               target_rate=config['loss'].get('target_rate', 0.05))

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
    epochs = config['training']['phase_3_epochs']
    accumulation_steps = config['training'].get('accumulation_steps', 1)
    
    optimizer = create_optimizer(model, loss_fn, config, low_encoder_lr=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    if resume:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['accuracy']
        best_dice = checkpoint['dice']
        print(f"Resuming training from epoch {start_epoch}...")
    
    training_loop(3, start_epoch, epochs, best_acc, best_dice, model, device, train_loader, val_loader, loss_fn, optimizer, scheduler, writer, checkpoint_dir)

    
phase_handles = {
    1: phase_one,
    2: phase_two,
    3: phase_three,
}


def run_training(config, model, device, phase, resume, loso, checkpoint=None):

    run_name = f"{config['experiment_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = os.path.join("saved_models", f"phase{phase}", run_name)
    
    
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
   
    train_set = SWEEPDataset(
        config, 
        split='train',
        experiment=True,
        loso=loso
    )
    
    val_set = SWEEPDataset(
        config, 
        split='val',
        experiment=True,
        loso=loso
    )

    train_loader = DataLoader(train_set, 
                              batch_size=config['training']['batch_size'],
                              shuffle=True, 
                              num_workers=config['data'].get('num_workers', 0),
                              prefetch_factor=4,
                              persistent_workers=True,
                              pin_memory=True)
    
    val_loader = DataLoader(val_set, 
                            batch_size=config['training']['batch_size'], 
                            shuffle=False, 
                            num_workers=config['data'].get('num_workers', 0),
                            prefetch_factor=4,
                            persistent_workers=True,
                            pin_memory=True)

    print(f"Data Loaded: {len(train_set)} Train | {len(val_set)} Val")
    
    model = model.to(device)
    phase_handles[phase](config, model, device, train_loader, val_loader, writer, checkpoint_dir, resume, checkpoint)
   
    print("--- Training Complete ---")
    wandb.finish()
    writer.close()