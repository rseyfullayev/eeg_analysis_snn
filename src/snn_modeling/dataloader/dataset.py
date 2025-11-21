import torch
import os
import numpy as np
from torch.utils.data import Dataset
from scipy.interpolate import griddata
from tqdm import tqdm 

class TopoMapper:
    def __init__(self, sensor_coords_df, grid_size=64, method='cubic'): # Using Azimuthal Equidistant Projection
        self.method = method
        self.grid_size = grid_size
        theta = sensor_coords_df['theta'].values
        r = sensor_coords_df['radius'].values
        self.x = r * np.cos(theta)
        self.y = r * np.sin(theta)
        self.points = np.column_stack((self.x, self.y))
        grid_x, grid_y = np.mgrid[-1:1:complex(0, grid_size), -1:1:complex(0, grid_size)]
        self.grid_z = (grid_x, grid_y)
        self.mask = (grid_x**2 + grid_y**2) > 1.0

    def transform(self, tensor):
        """
        Args:
            tensor: Shape (5, Time, 14) 
                    (Bands, Time Steps, Sensors)
            
        Returns:
            torch.Tensor: Shape (Time, 5, 64, 64)
        """
        if isinstance(tensor, torch.Tensor):
            data = tensor.numpy()
        else:
            data = tensor

        bands, time_steps, _ = data.shape
        output = np.zeros((time_steps, bands, self.grid_size, self.grid_size), dtype=np.float32)
        for t in range(time_steps):
            for b in range(bands):
                values = data[b, t, :]
                grid = griddata(self.points, values, self.grid_z, method=self.method, fill_value=0)
                grid[self.mask] = 0.0
                output[t, b, :, :] = grid
        
        return torch.tensor(output)
    
    def __call__(self, tensor):
        return self.transform(tensor)

class SWEEPDataset(Dataset):
    def __init__(self, config, mode='synthetic', data_folder=None, transform=None):
        self.config = config
        self.mode = mode
        self.transform = transform
        self.num_classes = config['model'].get('num_classes', 3)
        self.grid_size = config['data'].get('grid_size', 64)
        self.num_samples = config['data'].get('num_samples', 1)

        if self.mode == 'synthetic':
            self._init_synthetic_data()
        elif self.mode == 'real':
            raise ValueError(f"Unsupported mode: {self.mode}")
        elif self.mode == "load":
            if data_folder is None:
                raise ValueError("mode='load' requires 'data_folder' path.")
            self.load_from_disk(data_folder)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
    
    def _init_synthetic_data(self):
        T = self.config['data'].get('num_timesteps', 10)
        C = self.config['data'].get('freq_bands', 5)
        Ch = self.config['data'].get('num_channels', 14)
        print(f"Generating Synthetic EEG: {self.num_samples} samples, {Ch} channels, {T} time steps...")

        self.labels = torch.randint(0, self.num_classes, (self.num_samples,))
        self.raw_data = torch.randn(self.num_samples, C, T, Ch)

        for i in range(self.num_samples):
            label = self.labels[i]
            target_channel = label % Ch
            target_band = label % C
            self.raw_data[i, target_band, :, target_channel] += self.config['mask'].get('sigma', 5.0)
        
        if self.transform:
            print("   Applying TopoMapper to entire dataset...")
            transformed_list = []
            for i in tqdm(range(self.num_samples), desc="Topographic Mapping"):
                # Transform returns (Time, Bands, H, W)
                # We process one sample at a time
                vid = self.transform(self.raw_data[i]) 
                transformed_list.append(vid)
            
            # Stack into master tensor: (N, Time, Bands, H, W)
            self.data = torch.stack(transformed_list)
        else:
            # Fallback if no transform provided
            self.data = self.raw_data
        self._generate_prototypes()
        
    def _generate_prototypes(self):
        print("   Computing Data-Driven Prototypes...")
        prototypes = []

        for c in range(self.num_classes):
            class_data = self.data[self.labels == c]
            if len(class_data) == 0:
                print(f"   ⚠️ Warning: Class {c} has no samples. Using zeros.")
                prototype = torch.zeros(self.grid_size, self.grid_size)
            else:
                prototype = class_data.mean(dim=(0, 1, 2))
            
                p_min = prototype.min()
                p_max = prototype.max()
                if p_max > p_min:
                    prototype = (prototype - p_min) / (p_max - p_min)

            prototypes.append(prototype)

        self.prototypes = torch.stack(prototypes)
    
    def __len__(self):
        return self.num_samples
    
    def save_to_disk(self, folder_path):
        """Saves data, labels, and prototypes to .npy files."""
        os.makedirs(folder_path, exist_ok=True)
        
        print(f"💾 Saving dataset to {folder_path}...")
        np.save(os.path.join(folder_path, "data.npy"), self.data.cpu().numpy())
        np.save(os.path.join(folder_path, "labels.npy"), self.labels.cpu().numpy())
        np.save(os.path.join(folder_path, "prototypes.npy"), self.prototypes.cpu().numpy())
        print("   Save complete.")
    
    def load_from_disk(self, folder_path):
        """Loads data, labels, and prototypes from .npy files."""
        print(f"📂 Loading dataset from {folder_path}...")
        
        try:
            # Load as memory-mapped for speed (optional, but good for large data)
            # Use .copy() if you need to write to it, otherwise keep as read-only
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





