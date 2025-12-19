import torch
import os
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
import pandas as pd     
from sklearn.model_selection import train_test_split

class TopoMapper(nn.Module):
    def __init__(self, sensor_coords_df, grid_size=64, sigma=0.2, device='cuda'): # Using Azimuthal Equidistant Projection
        super(TopoMapper, self).__init__()
        self.sigma = sigma
        self.grid_size = grid_size
        theta = sensor_coords_df['theta'].values
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        if np.max(np.abs(theta)) > 2 * np.pi:
            print("   Detected DEGREES in coordinates. Converting to Radians.")
            theta = np.deg2rad(theta)
        else:
            print("   Detected RADIANS in coordinates.")
            theta = theta
        r = sensor_coords_df['radius'].values
       
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        x, y = y, x
        
        self.points = torch.tensor(np.column_stack((x, y)), dtype=torch.float32, device=self.device)

        grid_x, grid_y = torch.meshgrid(torch.linspace(-1, 1, grid_size, device=self.device),
                                        torch.linspace(-1, 1, grid_size, device=self.device), 
                                        indexing='ij')
        
        self.target_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        self.mask_indices = (self.target_points[:, 0]**2 + self.target_points[:, 1]**2) > 1.0
        dists = torch.cdist(self.target_points, self.points)
        weights = torch.exp(-(dists.pow(2)) / (2 * (self.sigma ** 2)))
        weights[self.mask_indices, :] = 0.0
        self.register_buffer('weights', weights)

    def transform(self, x):

        if x.device != self.weights.device:
            x = x.to(self.weights.device)
        
        
        original_shape = x.shape
        sensors = original_shape[-1]

        x_flat = x.reshape(-1, sensors)
        flat_images = torch.matmul(x_flat, self.weights.t())
        new_shape = original_shape[:-1] + (self.grid_size, self.grid_size)
        images = flat_images.view(new_shape)

        if len(original_shape) == 3:
             images = images.permute(1, 0, 2, 3)
        
        return images
    
    def __call__(self, tensor):
        return self.transform(tensor)

class SWEEPDataset(Dataset):
    def __init__(self, config, split='train', prototypes=None):
        self.config = config
        self.split = split
        self.num_classes = config['model'].get('num_classes', 5)
        self.grid_size = config['data'].get('grid_size', 32)
        self.dataset_path = config['data']['dataset_path'] 
        self.samples_dir = os.path.join(self.dataset_path)
        index_file = os.path.join(self.dataset_path, "index.csv")
        
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index not found at {index_file}. Did you run the baker?")
            
        print(f"Loading index from {index_file}...")
        df = pd.read_csv(index_file)
    

        indices = np.arange(len(df))
        labels = df['emotion_id'].values
        
        train_idx, val_idx = train_test_split(
            indices, 
            test_size=(1 - config['data'].get('train_split', 0.8)), 
            random_state=42, 
            stratify=labels # Ensures balanced classes in both sets
        )

        if split == 'train':
            print(f"Selecting TRAINING set ({len(train_idx)} samples)")
            df_slice = df.iloc[train_idx]
        elif split == 'val':
            print(f"Selecting VALIDATION set ({len(val_idx)} samples)")
            df_slice = df.iloc[val_idx]
        else:
            raise ValueError(f"Unknown split '{split}'. Use 'train' or 'val'.")
        self.samples = list(zip(df_slice['filename'], df_slice['emotion_id']))
        sigma = config['mask'].get('sigma', 0.25)
        radius = config['mask'].get('radius', 0.7)
        if prototypes is not None:
            self.prototypes = prototypes
        else:
            self.prototypes = self.compute_prototypes(self.num_classes, self.grid_size, radius, sigma, device='cpu')

    @staticmethod
    def compute_prototypes(num_classes, grid_size, radius, sigma, device='cpu'):
        range_t = torch.linspace(-1, 1, grid_size, device=device)
        yy, xx = torch.meshgrid(range_t, range_t, indexing='ij')
        prototypes = []

        for c in range(num_classes):
            angle = (2 * np.pi * c) / num_classes
            cx = radius * np.sin(angle)
            cy = radius * np.cos(angle)
            dist_sq = (xx - cx)**2 + (yy - cy)**2
            blob = torch.exp(-dist_sq / (2 * sigma**2))
            blob = blob / blob.max()
            prototypes.append(blob)

        return torch.stack(prototypes)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label_idx = self.samples[idx]
        
        file_path = os.path.join(self.samples_dir, fname)
        try:
            video = torch.load(file_path) 
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            return torch.zeros(5, 32, 32, 32), torch.zeros(self.num_classes, 32, 32), 0
        
        P98_VAL = 2396.1880
        GAIN = 10.0  
        
        video = video / P98_VAL * GAIN
        video = torch.tanh(video) 
 
        target_volume = torch.zeros(self.num_classes, self.grid_size, self.grid_size)
        target_volume[label_idx] = self.prototypes[label_idx]
        
        return video, target_volume, label_idx