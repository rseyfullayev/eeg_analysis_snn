# CITE https://github.com/HobbitLong/SupContrast/tree/master

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.snn_modeling.utils.loss import FullHybridLoss, TopKClassificationLoss, ContrastiveLoss
from src.snn_modeling.dataloader.dataset import SWEEPDataset, PKSampler
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
from src.snn_modeling.utils.augmentations import VideoTemporalMasking, GaussianNoise, FrequencyDropout, SignalJitter, VideoRandomErasing
from src.snn_modeling.layers.neurons import ALIF
from src.snn_modeling.models.unet import SpikingResNetClassifier
from omegaconf import OmegaConf
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

    val_loop = tqdm(val_loader, desc=f"Validation", unit="batch")
    with torch.no_grad():
        for batch_idx, (inputs, targets, labels, _) in enumerate(val_loop):
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

            if only_classification:
                #_,C,H,W = outputs.shape #(outputs * criterion.class_loss.masks).sum(dim=(2, 3))
                loss = 0 #criterion(outputs, labels) #.unsqueeze(1).expand(-1, T).permute(1,0).reshape(-1).view(-1,1,1).expand(-1,4,4).long())
                #energy_logits = outputs.view(T,B,C,H,W).mean(dim=[0,3,4])
                return 0,0,0,0,0,0,0
            else:
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
            if not only_classification:
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
    
    # Cleanup tensors
    del inputs, targets, labels, outputs, probs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def freeze_bn_stats(module):
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
            module.eval()

def create_optimizer(model, loss_fn, config, low_encoder_lr=False):
    lr = config.training.get('learning_rate', 1e-3)
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
        if 'alpha' in name or 'beta' in name or 'slope' in name or 'decay' in name or 'gamma' in name or 'recurrent' in name:
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
        {'params': base_params, 'lr': lr, 'weight_decay': config.training.get('weight_decay', 1e-4)},
        {'params': encoder_params, 'lr': lr * 1e-2, 'weight_decay': config.training.get('weight_decay', 1e-4)},
        {'params': base_params_no_decay, 'lr': lr, 'weight_decay': 0.0},
        {'params': time_params, 'lr': lr * 0.5, 'weight_decay': 0.0},
        {'params': threshold_params, 'lr': lr * 1.0, 'weight_decay': 0.0}
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
    
    
    #temp_mix = TemporalMix()

    for epoch in range(start_epoch, epochs):
        model.train()

        if freeze_bn:
            model.encoder.apply(freeze_bn_stats)
        train_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Phase {phase} Epoch {epoch+1}/{epochs}", unit="batch")
        for batch_idx, batch in enumerate(train_loop):
            # Handle both augmented (5 items) and non-augmented (4 items) returns
            if len(batch) == 5:
                inp1, inp2, targets, targets_c, _ = batch
                if phase == 1:
                    # Contrastive: concatenate both views
                    inp1, inp2 = inp1.to(device), inp2.to(device)
                    inputs = torch.cat([inp1, inp2], dim=0)
                else:
                    # Non-contrastive phases: just use first view
                    inputs = inp1.to(device)
            else:
                inputs, targets, targets_c, _ = batch
                inputs = inputs.to(device)
                
            targets, targets_c = targets.to(device), targets_c.to(device)
            
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

            if phase == 1:
                f1, f2 = torch.split(outputs, [B//2, B//2], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                loss = loss_fn(features, targets_c) #.unsqueeze(1).expand(-1, T).permute(1,0).reshape(-1).view(-1,1,1).expand(-1,4,4).long())
            else:
                loss = loss_fn(outputs, targets, targets_c)
           
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if phase > 1:
                with torch.no_grad():
                    for name, param in model.decoder.named_parameters():
                        if 'threshold' in name:
                            param.clamp_(0.1, 3.0)
            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())
            
            # Explicit cleanup to prevent memory accumulation
            del inputs, targets_c, outputs, loss

        # Periodic memory cleanup after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        avg_train_loss = train_loss / len(train_loader)
          
        val_loss, val_acc, val_bal_acc, val_dice, val_iou, val_pre, val_rec = validate(model, val_loader, loss_fn, device, only_classification=phase == 1)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        save_checkpoint(model, optimizer, scheduler, epoch, best_acc, best_dice, f"{checkpoint_dir}/checkpoint_temp.pt") 
        os.replace(f"{checkpoint_dir}/checkpoint_temp.pt", f"{checkpoint_dir}/checkpoint_last.pt")
        
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

        
        for k, v in log_dict.items(): writer.add_scalar(k, v, epoch)

        if (-1 if phase==1 else 1)  * val_acc > (-1 if phase==1 else 1) * best_acc:
            best_acc = avg_train_loss if phase == 1 else val_acc
            best_dice = val_dice
            save_checkpoint(model, optimizer, scheduler, epoch, best_acc, best_dice, f"{checkpoint_dir}/checkpoint_best.pt") #_{epoch:03d}_{best_acc:.4f}_{best_dice:.4f}
            wandb.save(f"{checkpoint_dir}/checkpoint_best.pt")
        
        elif val_acc == best_acc and val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(model, optimizer, scheduler, epoch, best_acc, best_dice, f"{checkpoint_dir}/checkpoint_{epoch:03d}_{best_acc:.4f}_{best_dice:.4f}.pt")
        else:
            print(f"Best Result yet: {best_acc:.4f}")

        wandb.log(log_dict, step=epoch)


def phase_one(config, model, device, train_loader, val_loader, writer, checkpoint_dir, resume, checkpoint=None):
    print("=== Phase One: Training Encoder Only ===")
    
    if isinstance(model, SpikingResNetClassifier):
        enc_class = model
    else:
        # Fallback for old style config!
        enc_class = SpikingResNetClassifier(
            encoder_backbone = model.encoder,
            num_classes=config.model.get('num_classes', 5),
            use_swiglu=config.model.get('use_swiglu', False)
        ).to(device)

    #initialize_network(enc_class, train_loader, device)
    """loss_fn = FullHybridLoss(
        smooth = 0.,
        lambda_seg = config.loss.get('lambda_seg', 1.0),
        lambda_con = config.loss.get('lambda_con', 0.0),
        lambda_class = config.loss.get('lambda_class', 1.0),
        alpha = 0.,
        beta = 0.,
        time_steps=config.data.get('num_timesteps', 16),
    )"""

    loss_fn = ContrastiveLoss(train_loader.dataset.prototypes) #nn.CrossEntropyLoss(label_smoothing=0.1)

    loss_fn.to(device)

    start_epoch = 0
    best_acc = 100.0
    best_dice = 0.0
    epochs = config.training.phase_1_epochs
    warmup_epochs = config.training.get('warmup_epochs', 0)
    accumulation_steps = config.training.get('accumulation_steps', 1)
    optimizer = create_optimizer(enc_class, loss_fn, config, low_encoder_lr=False)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
    '''
    SequentialLR(optimizer, [
        LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
        CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
    ], milestones=[warmup_epochs])
    '''
    
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
        GaussianNoise(std=0.01),
        FrequencyDropout(p=0.1),
        VideoTemporalMasking(p=0.1, max_mask_len=2),
        #SignalJitter(lower=0.8, upper=1.2),
        #VideoRandomErasing(p=0.3)
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
    
    train_loader = DataLoader(train_set, 
                              batch_sampler=PKSampler(train_set, 
                                                      batch_size=config.training.batch_size, 
                                                      n_classes=config.model.get('n_emotions', 5)),
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