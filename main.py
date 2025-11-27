import yaml
import argparse
import torch
from src.snn_modeling.utils.model_builder import build_model
from train import run_training
from setup_data import setup_data



def main(config_path, run_setup_data=False, checkpoint_path=None):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    if run_setup_data:
        print("Running dataset setup...")
        setup_data(config)
    else:
        checkpoint = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            print(f"Loaded checkpoint from {checkpoint_path}.")
        
        model = build_model(config).to(device)
        run_training(config, model, device, checkpoint)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--mode', type=str, default='train', help='Mode: train or test')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint')
    #Check if --setup_data flag is provided, run setup_data function
    parser.add_argument('--setup_data', action='store_true', help='Setup data before training')

    args = parser.parse_args()
    
    main(args.config, args.setup_data, args.checkpoint)