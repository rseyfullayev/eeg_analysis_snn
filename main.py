import argparse
import yaml
import os
import torch
import torch.nn as nn

from generate_dataset import run_data_setup
from train import run_training, validate
from src.snn_modeling.models.unet import SpikingResNetClassifier
from src.snn_modeling.utils.model_builder import build_model
from src.snn_modeling.utils.utils import run_bio_audit, calculate_optimal_firing_rate, analyze_distribution, seed_everything, generate_topology_proof, find_representative_subject, generate_masks, calibrate_params
from src.snn_modeling.dataloader.dataset import SWEEPDataset
from src.snn_modeling.utils.loss import FullHybridLoss

from torch.utils.data import DataLoader

def main():
    seed_everything(42)
    parser = argparse.ArgumentParser(description="SWEEP-Net Entry Point")

    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--mode', type=str, default='train', help='Mode: train or test')
    parser.add_argument('--loso', type=int, help='The integer ID of the subject to hold out for testing (1-16).')
    parser.add_argument('--subj', type=int, help='The integer ID of the subject (1-16).')

    parser.add_argument('--phase', type=int, help='Specify training phase (1, 2, 3, or 4)')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--val', action='store_true', help='Validation mode')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint')

    parser.add_argument('--setup_data', action='store_true', help='Setup data before training')

    parser.add_argument('--raw_path', type=str, help='Path to folder containing raw .txt files')
    parser.add_argument('--coords_path', type=str, help='Path to electrodes coordinates .csv')
    parser.add_argument('--output_path', type=str, help='Destination folder for processed .npy files')

    parser.add_argument('--calculate_stat', action='store_true', help='Draw Distribution of dataset')
    parser.add_argument('--find_repr', action='store_true', help='Find the most representative subject in the dataset')
    parser.add_argument('--audit_bio', action='store_true', help='Run Biological Audit (Model-Free)')
    parser.add_argument('--masks', type=int, help='Derive masks from Subject')
    parser.add_argument('--calibrate', action='store_true', help='Calibrate optimal ALIF parameters')
    args = parser.parse_args()
    config_path = args.config

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    if args.resume and args.checkpoint is None:
        parser.error("When using --resume, you MUST specify --checkpoint.")
    
    if args.loso and args.subj:
        parser.error("You CANNOT specify both --loso and --subj at the same time.")
    
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)


    if args.audit_bio:
        run_bio_audit(config, device=torch.device('cuda'), samples=300)

    elif args.calibrate:
        if args.phase != 2:
            parser.error("You MUST specify --phase 2 for calibration.")
        if args.checkpoint is None:
            parser.error("You MUST specify --checkpoint for calibration.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(config).to(device)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        print(f"Loaded checkpoint from {args.checkpoint}.")
        chk = {k[8:]:v for k,v in checkpoint['model_state_dict'].items() if 'encoder' in k}
        model.encoder.load_state_dict(chk)

        masks = torch.load(os.path.join(config['data']['dataset_path'],'masks.pt')).to(device)
        val_set = SWEEPDataset(
                                config, 
                                split='val',
                                experiment=False,
                                loso=args.loso,
                                subj=args.subj,
                                prototypes=masks
                                )
        
        val_loader = DataLoader(val_set, 
                            batch_size=config['training']['batch_size'], 
                            shuffle=False, 
                            num_workers=config['data'].get('num_workers', 0),
                            prefetch_factor=4,
                            persistent_workers=True,
                            pin_memory=True)
        
        
        calibrate_params(model.encoder, val_loader, device)
    
    elif args.masks is not None:
        generate_masks(config, subject_id=args.masks)

    elif args.calculate_stat:
        dataset = SWEEPDataset(
                config, 
                split='train',
                loso=0, 
                )   
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        print("Calculating Dataset Statistics...")

        analyze_distribution(dataloader)
        #calculate_optimal_firing_rate(dataset)
        generate_topology_proof(dataloader, torch.device("cuda"), class_names=[0,1,2,3,4])

        

    elif args.setup_data:
        print("Running dataset setup...")
        if not args.raw_path or not args.coords_path or not args.output_path:
            parser.error("When using --setup_data, you MUST specify --raw_path, --coords_path, and --output_path.")
        config['data']['raw_path'] = args.raw_path
        config['data']['coords_path'] = args.coords_path
        config['data']['dataset_path'] = args.output_path
        print(f"   Raw Source: {args.raw_path}")
        print(f"   Coordinates: {args.coords_path}")
        print(f"   Target: {args.output_path}")
        
        # Execute Setup
        run_data_setup(config)
        
        print("Setup complete.")
    elif args.val:
        checkpoint = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        if args.checkpoint is not None:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            print(f"Loaded checkpoint from {args.checkpoint}.")
        
        model = build_model(config).to(device)
        if args.phase == 1:
            model = SpikingResNetClassifier(
                                            encoder_backbone = model.encoder,
                                            num_classes=config['model'].get('num_classes', 5)
                                            ).to(device)
        if checkpoint is not None:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("No checkpoint provided; using untrained model.")

        if not args.phase:
            parser.error("You MUST specify --phase for validation.")
        
        if not args.loso and not args.subj:
            parser.error("You MUST specify --loso or --subj for validation.")
        
        masks = torch.load(os.path.join(config['data']['dataset_path'],'masks.pt')).to(device)
        val_set = SWEEPDataset(
                                config, 
                                split='val',
                                experiment=False,
                                loso=args.loso,
                                subj=args.subj,
                                prototypes=masks
                                )
        
        val_loader = DataLoader(val_set, 
                            batch_size=config['training']['batch_size'], 
                            shuffle=False, 
                            num_workers=config['data'].get('num_workers', 0),
                            prefetch_factor=4,
                            persistent_workers=True,
                            pin_memory=True)
        print(f"Validation set size: {len(val_set)} samples.")

        loss_fn = FullHybridLoss(
        smooth = 0.0,
        lambda_seg = config['loss'].get('lambda_seg', 1.0),
        lambda_con = 0.0,
        lambda_class = config['loss'].get('lambda_class', 1.0),
        alpha = config['loss'].get('alpha', 0.5),
        beta = config['loss'].get('beta', 0.5),
        time_steps=config['data'].get('num_timesteps', 16),
        )

        loss_fn.class_loss.masks = val_loader.dataset.prototypes

        if args.phase == 1: loss_fn = nn.CrossEntropyLoss()

        val_loss, val_acc, val_bal_acc, val_dice, val_iou, val_pre, val_rec = validate(model, val_loader, loss_fn, device, only_classification=args.phase==1)
        if args.phase == 1:
            print(f"Phase {args.phase} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        else:
            print(f"Phase {args.phase} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Val Acc: {val_acc:.4f} | Dice: {val_dice:.4f} | "
                  f"Val Pre: {val_pre:.4f} | Val Rec: {val_rec:.4f}")

    else:
        checkpoint = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        if args.checkpoint is not None:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            print(f"Loaded checkpoint from {args.checkpoint}.")
        
        model = build_model(config).to(device)

        if args.find_repr:
            enc_class = SpikingResNetClassifier(
            encoder_backbone = model.encoder,
            num_classes=config['model'].get('num_classes', 5)
            ).to(device)
            
            enc_class.load_state_dict(checkpoint['model_state_dict'])
            find_representative_subject(enc_class, config, device, samples_per_subject=500)
            
            exit(0)
        
        if not args.phase:
            parser.error("You MUST specify --phase for training/testing.")
        
        if not args.loso and not args.subj:
            parser.error("You MUST specify --loso or --subj for training/testing.")
        
        
        run_training(config, model, device, phase=args.phase, resume=args.resume, loso=args.loso, subj=args.subj, checkpoint=checkpoint)

if __name__ == "__main__":
    main()