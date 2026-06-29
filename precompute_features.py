import argparse
import os
import torch
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose

from src.snn_modeling.utils.model_builder import build_model
from src.snn_modeling.dataloader.dataset import SWEEPDataset
from tqdm import tqdm

def run_precompute(config, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    print("Building model...")
    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # We only care about the encoder. We can load safely.
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    encoder = model.encoder
    encoder.eval()
    
    # Create dataset (we don't bag here, we just want to get all windows grouped by bag_id)
    dataset = SWEEPDataset(config, split='train', precomputed=False)
    
    precomputed_dir = os.path.join(config.data.dataset_path, "precomputed_features")
    os.makedirs(precomputed_dir, exist_ok=True)
    
    # We will process bags one by one
    print(f"Precomputing features for {len(dataset.samples)} bags...")
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset))):
            bag_id, files, _, _, _, _ = dataset.samples[idx]
            
            # Load windows for this bag
            loaded_tensors = []
            for fname in files:
                file_path = os.path.join(dataset.samples_dir, fname)
                try:
                    video = torch.load(file_path, weights_only=True, map_location='cpu').float()
                    loaded_tensors.append(video)
                except Exception as e:
                    print(f"Error loading {fname}: {e}")
            
            if not loaded_tensors:
                continue
                
            # (W, C, H, W_t, T) -> Assuming 5D tensors from generate_dataset
            bag_video = torch.stack(loaded_tensors, dim=0).to(device)
            
            # Run through encoder
            features, _ = encoder(bag_video)
            
            if features.dim() == 5:
                out = features.mean(dim=[0, 3, 4])  # mean over T, H, W
            else:
                out = features.mean(dim=[-2, -1])   # mean over H, W if already collapsed
            
            # out is now (W, C_dim)
            out = out.cpu()
            
            save_path = os.path.join(precomputed_dir, f"bag_{bag_id}.pt")
            torch.save(out, save_path)
            
    print("Precomputation complete!")

def main():
    parser = argparse.ArgumentParser(description="Precompute Phase 1A features")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to Phase 1A checkpoint')
    
    args = parser.parse_args()
    
    # Setup config
    config_dir = os.path.abspath(os.path.dirname(args.config))
    config_name = os.path.basename(args.config).replace('.yaml', '')
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        config = compose(config_name=config_name)
    
    from omegaconf import open_dict
    with open_dict(config):
        if hasattr(config, 'architecture'):
            config.model = config.architecture
            
    run_precompute(config, args.checkpoint)

if __name__ == "__main__":
    main()
