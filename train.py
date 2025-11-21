import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from src.snn_modeling.utils.loss import FullHybridLoss
from src.snn_modeling.dataloader.dummy_loader import get_dummy_batch
from src.snn_modeling.dataloader.dataset import SWEEPDataset
import os
from datetime import datetime

from sklearn.metrics import accuracy_score, balanced_accuracy_score
import snntorch as snn
import segmentation_models_pytorch as smp
import wandb
import warnings
from sklearn.exceptions import UndefinedMetricWarning
# Ignore the specific sklearn warning about missing classes
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
warnings.filterwarnings("ignore", message="A single label was found in 'y_true' and 'y_pred'. For the confusion matrix to have the correct shape, use the 'labels' parameter to pass all known labels.")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
def manual_reset(model):
    """
    Recursively reset the hidden state of all SNN layers.
    This ensures the computation graph is broken between epochs.
    """
    for module in model.modules():
        # Check if the module is a spiking neuron
        if isinstance(module, (snn.Leaky, snn.Synaptic, snn.Alpha)):
            if hasattr(module, 'reset_mem'):
                module.reset_mem()
            if hasattr(module, 'reset_hidden'):
                module.reset_hidden()
            #if hasattr(module, 'detach_hidden'):
            #    module.detach_hidden()

def save_checkpoint(model, optimizer, epoch, acc, dice, path="best_sweepnet.pt"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': acc,
        'dice': dice
    }, path)
    print(f"New Record! Saved checkpoint at Epoch {epoch} (Acc: {acc:.4f}, Dice: {dice:.4f})")

def validate(model, val_loader, criterion, device, config):
    """
    Validation Loop. Calculates generic metrics over the unseen dataset.
    """
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    # Accumulators for Sklearn metrics
    all_preds = []
    all_targets = []
    
    # Accumulators for Segmentation metrics
    tp_tot, fp_tot, fn_tot, tn_tot = 0, 0, 0, 0

    with torch.no_grad():
        for inputs, targets, labels in val_loader:
            inputs, targets, labels = inputs.to(device), targets.to(device), labels.to(device)
            
            # SNN Input Format: (Time, Batch, Channels, H, W)
            inputs = inputs.permute(1, 0, 2, 3, 4)
            
            manual_reset(model) # Crucial: Reset before validation batch
            
            # Forward
            outputs = model(inputs) # (Time, Batch, 5, H, W)
            outputs_avg = torch.mean(outputs, dim=0) # (Batch, 5, H, W)
            
            # Loss
            loss = criterion(outputs_avg, targets, labels)
            val_loss += loss.item()
            
            # --- 1. Classification Metrics (Energy) ---
            # Use Soft Probabilities (Sigmoid) for Energy calculation
            soft_probs = torch.sigmoid(outputs_avg)
            energy = torch.sum(soft_probs, dim=(2, 3)) # Sum over H, W -> (Batch, 5)
            preds = torch.argmax(energy, dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

            # --- 2. Segmentation Metrics (Dice) ---
            # Binarize for visual/overlap metrics
            pred_masks = (soft_probs > 0.5).long()
            target_masks = (targets > 0.5).long()
            
            # Get stats batch-wise
            tp, fp, fn, tn = smp.metrics.get_stats(
                pred_masks, target_masks, mode='multilabel', threshold=0.5
            )
            tp_tot += tp.sum().item()
            fp_tot += fp.sum().item()
            fn_tot += fn.sum().item()
            tn_tot += tn.sum().item()

    # --- Compute Epoch Metrics ---
    avg_loss = val_loss / len(val_loader)
    accuracy = correct / total
    balanced_acc = balanced_accuracy_score(all_targets, all_preds)
    
    eps = 1e-7
    dice_score = (2 * tp_tot) / (2 * tp_tot + fp_tot + fn_tot + eps)
    iou_score = tp_tot / (tp_tot + fp_tot + fn_tot + eps)
    precision = tp_tot / (tp_tot + fp_tot + eps)
    recall = tp_tot / (tp_tot + fn_tot + eps)

    return avg_loss, accuracy, balanced_acc, dice_score, iou_score, precision, recall

def run_training(config, model, device):

    run_name = f"{config['experiment_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project=config['logging']['project_name'],
        name=config['logging']['run_name'],
        config=config,
        tags=config['logging']['tags'],
        mode="disabled" if config['logging'].get('offline') else "online"
    )
    
    log_dir = os.path.join("results", run_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"Initializing TensorBoard: {log_dir}")

    full_dataset = SWEEPDataset(
        config, 
        mode='load', 
        data_folder=config['data']['dataset_path']
    )
    
    train_size = int(config['data'].get('train_split', 0.8) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['data'].get('num_workers', 0))
    val_loader = DataLoader(val_set, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['data'].get('num_workers', 0))
    print(f"Data Loaded: {len(train_set)} Train | {len(val_set)} Val")

    model = model.to(device)
    model.train()
 
    lr = config['training'].get('learning_rate', 1e-3)
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=config['training'].get('weight_decay', 1e-5))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

    loss_fn = FullHybridLoss(
        smooth = config['loss'].get('smooth', 0.0),
        lambda_bce = config['loss'].get('lambda_bce', 1.0),
        lambda_class = config['loss'].get('lambda_class', 1.0)
    )
    
    epochs = config['training']['epochs']
    best_acc = 0.0
    best_dice = 0.0


    for epoch in range(epochs):
        
        model.train()
        train_loss = 0.0

        for batch_idx, (inputs, targets, targets_c) in enumerate(train_loader):
            inputs, targets, targets_c = inputs.to(device), targets.to(device), targets_c.to(device)
            inputs = inputs.permute(1, 0, 2, 3, 4)

            manual_reset(model)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            outputs_agg = torch.mean(outputs, dim=0)

            loss = loss_fn(outputs_agg, targets, targets_c)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        val_loss, val_acc, val_bal_acc, val_dice, val_iou, val_pre, val_rec = validate(model, val_loader, loss_fn, device, config)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1} | LR: {current_lr:.2e} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | Dice: {val_dice:.4f}")
        
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
            pass 

        if val_acc > best_acc:
            best_acc = val_acc
            best_dice = val_dice
            save_checkpoint(model, optimizer, epoch, best_acc, best_dice, run_name)
        elif val_acc == best_acc and val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(model, optimizer, epoch, best_acc, best_dice, run_name)
            
    
    print("--- Training Complete ---")
    wandb.finish()
    writer.close()