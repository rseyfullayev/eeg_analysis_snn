import torch
import os
import json
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
import pandas as pd     
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from torch.utils.data.sampler import Sampler
import random
import collections

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

        r = r / (np.max(np.abs(r)) + 1e-8)
       
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
        weight_sums = weights.sum(dim=1, keepdim=True)
        weight_sums = torch.clamp(weight_sums, min=1e-8)
        weights = weights / weight_sums
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
    def __init__(self, config, loso=None, subj=None, split='train', experiment=False, prototypes=None, augmentations=None):
        self.config = config
        self.split = split
        self.num_classes = config['data'].get('n_emotions', 5)
        self.grid_size = config['data'].get('grid_size', 32)
        self.dataset_path = config['data']['dataset_path'] 
        self.samples_dir = os.path.join(self.dataset_path)
        self.preload = config['data'].get('preload_ram', False)
        self.train_size = config['data'].get('train_size', 0.8)
        self.augmentations = augmentations
        self.cache = {}

        index_file = os.path.join(self.dataset_path, "index.csv")
        stats_path = os.path.join(self.dataset_path, "stats.json")

        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index not found at {index_file}.")
            
        print(f"Loading index from {index_file}...")
        df = pd.read_csv(index_file)

        with open(stats_path, 'r') as f:
            self.stats_lookup = json.load(f)

        indices = np.arange(len(df))
        labels = df['emotion_id'].values
        if subj is not None:

            df = df[df['filename'].str.split('_').str[0] == str(subj)]
            #df_train, df_val = train_test_split(df, test_size=0.2, random_state=42, stratify=df['emotion_id'])
            splitter = GroupShuffleSplit(n_splits=1, test_size=1.0 - self.train_size, random_state=42)
            train_idx, val_idx = next(splitter.split(df, groups=df['group_id']))

            df_train = df.iloc[train_idx]
            df_val = df.iloc[val_idx]

        else:
            df_train = df[df['filename'].str.split('_').str[0] != str(loso)]
            df_val = df[df['filename'].str.split('_').str[0] == str(loso)]
    

        if split == 'train':
            print(f"Selecting TRAINING set ({len(df_train)} samples)")
            df_slice = df_train if not experiment else df_train.sample(25000, random_state=42) #[:25000]
        elif split == 'val':
            print(f"Selecting VALIDATION set ({len(df_val)} samples)")
            df_slice = df_val if not experiment else df_val.sample(6400, random_state=42)#[:6400]
        else:
            raise ValueError(f"Unknown split '{split}'. Use 'train' or 'val'.")
        
        self.samples = list(zip(df_slice['filename'], df_slice['emotion_id'], df_slice['stats_key']))

        if self.preload:
            print("Preloading data into RAM...")
            for fname, _ in self.samples:
                file_path = os.path.join(self.samples_dir, fname)
                try:
                    # Use weights_only=True for security, map_location='cpu' to avoid GPU memory
                    # Use half precision to reduce RAM by 50% if acceptable
                    video = torch.load(file_path, weights_only=True, map_location='cpu').float()
                    # Share memory for multiprocessing efficiency
                    video.share_memory_()
                    self.cache[fname] = video
                except Exception as e:
                    print(f"Error loading {fname}: {e}")
            print("Cache complete!")


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
        fname, label_idx, stats_key = self.samples[idx]

        #mean = self.stats_lookup[stats_key]['mean']
        #std = self.stats_lookup[stats_key]['std']

        
        file_path = os.path.join(self.samples_dir, fname)
        if self.preload:
            video = self.cache[fname]  # No clone needed - data is not modified in-place
        else:
            try:
                video = torch.load(file_path, weights_only=True, map_location='cpu').float()
            except Exception as e:
                print(f"Error loading {fname}: {e}")
                return torch.zeros(5, 32, 32, 32), torch.zeros(32, 32), 0
        
        mean = video.mean(dim=(2,3,4), keepdim=True)
        std = video.std(dim=(2,3,4), keepdim=True) + 1e-8


        video = (video - mean) / (std)
        target_map = torch.zeros((self.grid_size, self.grid_size), dtype=torch.long)
        target_map[self.prototypes[label_idx] > 0.1] = label_idx + 1  # Background is 0
        if self.augmentations is not None and self.split == 'train':
            video1 = self.augmentations(video)
            video2 = self.augmentations(video)
            return video1, video2, target_map, label_idx, fname
        
        return video, target_map, label_idx, fname
    

class PKSampler(Sampler):
    def __init__(self, dataset, batch_size, n_classes=5, n_samples_per_class=None):
        self.labels = [s[2] for s in dataset]
        self.labels = torch.tensor(self.labels).long()
        self.label_set = list(set(self.labels.numpy()))
        
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.label_set}
        
        for l in self.label_set:
            np.random.shuffle(self.label_to_indices[l])
            
        self.used_label_indices_count = {label: 0 for label in self.label_set}
        self.count = 0
        self.n_classes = n_classes
        
        if n_samples_per_class is None:
            self.n_samples_per_class = batch_size // n_classes
        else:
            self.n_samples_per_class = n_samples_per_class
            
        self.batch_size = self.n_samples_per_class * self.n_classes
        self.dataset_len = len(dataset)

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.dataset_len:
            classes = np.random.choice(self.label_set, self.n_classes, replace=False)
            indices = []
            
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:
                               self.used_label_indices_count[class_] + self.n_samples_per_class])
                
                self.used_label_indices_count[class_] += self.n_samples_per_class
                
                if self.used_label_indices_count[class_] + self.n_samples_per_class > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
                    
            yield indices
            self.count += self.batch_size

    def __len__(self):
        return self.dataset_len // self.batch_size