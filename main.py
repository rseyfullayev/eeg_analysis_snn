import argparse
import yaml
import os
import torch

from generate_dataset import run_data_setup
from train import run_training
from src.snn_modeling.models.unet import SpikingResNetClassifier
from src.snn_modeling.utils.model_builder import build_model
from src.snn_modeling.utils.utils import calculate_p98, calculate_optimal_firing_rate, analyze_distribution, seed_everything, generate_topology_proof, find_representative_subject
from src.snn_modeling.dataloader.dataset import SWEEPDataset

from torch.utils.data import DataLoader

def main():
    seed_everything(42)
    parser = argparse.ArgumentParser(description="SWEEP-Net Entry Point")

    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--mode', type=str, default='train', help='Mode: train or test')
    parser.add_argument('--loso', type=int, help='The integer ID of the subject to hold out for testing (1-16).')

    parser.add_argument('--phase', type=int, help='Specify training phase (1, 2, 3, or 4)')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint')

    parser.add_argument('--setup_data', action='store_true', help='Setup data before training')

    parser.add_argument('--raw_path', type=str, help='Path to folder containing raw .txt files')
    parser.add_argument('--coords_path', type=str, help='Path to electrodes coordinates .csv')
    parser.add_argument('--output_path', type=str, help='Destination folder for processed .npy files')

    parser.add_argument('--calculate_stat', action='store_true', help='Calculate Fire Rate and Draw Distribution of dataset')
    parser.add_argument('--find_repr', action='store_true', help='Find the most representative subject in the dataset')

    args = parser.parse_args()
    config_path = args.config

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    if args.resume and args.checkpoint is None:
        parser.error("When using --resume, you MUST specify --checkpoint.")
    
    
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    if args.calculate_stat:
        dataset = SWEEPDataset(
                config, 
                split='train',
                loso=0, 
                )   
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        print("Calculating Dataset Statistics...")

        analyze_distribution(dataloader)
        calculate_optimal_firing_rate(dataset)
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
        
        if not args.loso:
            parser.error("You MUST specify --loso for training/testing.")
        
        
        run_training(config, model, device, phase=args.phase, resume=args.resume, loso=args.loso, checkpoint=checkpoint)

if __name__ == "__main__":
    main()