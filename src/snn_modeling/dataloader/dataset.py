import torch
import os
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
from scipy.interpolate import griddata, RBFInterpolator

from tqdm import tqdm 

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

        x_flat = x.view(-1, sensors)
        flat_images = torch.matmul(x_flat, self.weights.t())
        new_shape = original_shape[:-1] + (self.grid_size, self.grid_size)
        images = flat_images.view(new_shape)

        if len(original_shape) == 3:
             images = images.permute(1, 0, 2, 3)
        
        return images
    
    def __call__(self, tensor):
        return self.transform(tensor)

class SWEEPDataset(Dataset):
    def __init__(self, config, data=None, labels=None, prototypes=None, mode='manual'):
        self.config = config
        self.mode = mode
        self.num_classes = config['model'].get('num_classes', 5)
        self.grid_size = config['data'].get('grid_size', 64)

        if self.mode == 'manual':
            self.data = data
            self.labels = labels
        elif self.mode == 'load':
            self.load_from_disk(config['data']['dataset_path'])
        if prototypes is not None:
            self.prototypes = prototypes
        else:
            print("Generating prototypes from local data (Ensure this is Train set!)")
            self.prototypes = self._generate_prototypes_internal()
    
    def _generate_prototypes_internal(self):
        return SWEEPDataset.compute_prototypes(self.data, self.labels, self.num_classes, self.grid_size)
    
    @staticmethod
    def compute_prototypes(num_classes, grid_size, device='cuda'):
        range_t = torch.linspace(-1, 1, grid_size, device=device)
        yy, xx = torch.meshgrid(range_t, range_t, indexing='ij')
        prototypes = []
        radius = 0.7
        sigma = 0.25
        
        for c in range(num_classes):
            angle = (2 * np.pi * c) / num_classes
            
            cx = radius * np.sin(angle)
            cy = radius * np.cos(angle)
            
            dist_sq = (xx - cx)**2 + (yy - cy)**2
            blob = torch.exp(-dist_sq / (2 * sigma**2))

            blob = blob / blob.max()
            prototypes.append(blob)

        return prototypes

    def __len__(self):
        return len(self.data)
    
    def save_to_disk(self, folder_path):
        """Saves data, labels, and prototypes to .npy files."""
        os.makedirs(folder_path, exist_ok=True)
        
        print(f"Saving dataset to {folder_path}...")
        np.save(os.path.join(folder_path, "data.npy"), self.data.cpu().numpy())
        np.save(os.path.join(folder_path, "labels.npy"), self.labels.cpu().numpy())
        np.save(os.path.join(folder_path, "prototypes.npy"), self.prototypes.cpu().numpy())
        print("   Save complete.")
    
    def load_from_disk(self, folder_path):
        """Loads data, labels, and prototypes from .npy files."""
        print(f"Loading dataset from {folder_path}...")
        
        try:
            self.data = torch.tensor(np.load(os.path.join(folder_path, "data.npy")), dtype=torch.float32)
            self.labels = torch.tensor(np.load(os.path.join(folder_path, "labels.npy")), dtype=torch.long)
            self.prototypes = torch.tensor(np.load(os.path.join(folder_path, "prototypes.npy")), dtype=torch.float32)
            
            self.num_samples = len(self.labels)
            print(f"   Loaded {self.num_samples} samples successfully.")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not find dataset files in {folder_path}. ensure data.npy, labels.npy, and prototypes.npy exist.") from e
    
    def __getitem__(self, idx):
        video = self.data[idx]
        label_idx = self.labels[idx]

        P98_VAL = 3758.30126953125 # The Holy Number
        GAIN = 10.0  # The gain used in the tanh
        video = np.tanh(video / P98_VAL * GAIN)
        # 4. Tensor Conversion
        video = torch.tensor(video, dtype=torch.float32)

        # Target Logic...
        target_volume = torch.zeros(self.num_classes, self.grid_size, self.grid_size)
        target_volume[label_idx] = self.prototypes[label_idx]
        
        return video, target_volume, label_idx





