import torch
import os
import numpy as np
from torch.utils.data import Dataset
from scipy.interpolate import griddata, RBFInterpolator

from tqdm import tqdm 

class TopoMapper:
    def __init__(self, sensor_coords_df, grid_size=64, method='thin_plate_spline'): # Using Azimuthal Equidistant Projection
        self.method = method
        self.grid_size = grid_size
        theta = sensor_coords_df['theta'].values
        if np.max(np.abs(theta)) > 2 * np.pi:
            print("   Detected DEGREES in coordinates. Converting to Radians.")
            theta = np.deg2rad(theta)
        else:
            print("   Detected RADIANS in coordinates.")
            theta = theta
        r = sensor_coords_df['radius'].values
       
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        x, y= y, x
        
        self.points = np.column_stack((x, y))
        grid_x, grid_y = np.mgrid[-1:1:complex(0, grid_size), -1:1:complex(0, grid_size)] # trick to generate grid_size points without floating point issues
        self.target_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        self.mask_indices = (self.target_points[:, 0]**2 + self.target_points[:, 1]**2) > 1.0

        self.kernel = method

    def transform(self, tensor):

        if isinstance(tensor, torch.Tensor):
            data = tensor.numpy()
        else:
            data = tensor

        bands, time_steps, sensors  = data.shape
        data_transposed = np.transpose(data, (2, 0, 1))

        flat_data = data_transposed.reshape(sensors, -1)

        interpolator = RBFInterpolator(self.points, flat_data, kernel=self.kernel)

        flat_maps = interpolator(self.target_points)
        flat_maps = flat_maps.T
        flat_maps[:, self.mask_indices] = 0.0
        images = flat_maps.reshape(bands, time_steps, self.grid_size, self.grid_size)
        images = images.transpose(1, 0, 2, 3)
        
        return torch.tensor(images, dtype=torch.float32)
    
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
    def compute_prototypes(data_tensor, label_tensor, num_classes, grid_size):

        print("   Computing Data-Driven Prototypes (Train Set Only)...")
        prototypes = []
        
        for c in range(num_classes):
            class_indices = (label_tensor == c).nonzero(as_tuple=True)[0]
            class_data = data_tensor[class_indices]
            
            if len(class_data) == 0:
                print(f"    Warning: Class {c} missing in training set. Using zeros.")
                prototype = torch.zeros(grid_size, grid_size)
            else:
                prototype = class_data.mean(dim=(0, 1, 2))

                p_min, p_max = prototype.min(), prototype.max()
                if p_max > p_min:
                    prototype = (prototype - p_min) / (p_max - p_min)
            
            prototypes.append(prototype)
            
        return torch.stack(prototypes)
    
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

        target_volume = torch.zeros(self.num_classes, self.grid_size, self.grid_size)
 
        target_volume[label_idx] = self.prototypes[label_idx]
        
        return video, target_volume, label_idx





