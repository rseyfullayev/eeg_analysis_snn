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
    def __init__(self, sensor_coords_df, grid_size=64, perplexity=5.0, device='cuda'):  # t-SNE binary search with perplexity
        super(TopoMapper, self).__init__()
        self.perplexity = perplexity
        self.grid_size = grid_size
        theta = sensor_coords_df['theta'].values
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        if np.max(np.abs(theta)) > 2 * np.pi:
            print("   Detected DEGREES in coordinates. Converting to Radians.")
            theta = np.deg2rad(theta)
        else:
            print("   Detected RADIANS in coordinates.")
            theta = theta
        r = sensor_coords_df['radius'].values.astype(np.float32)

        r = r / (np.max(np.abs(r)) + 1e-8)
        r = r * 0.95  # Scale to fit within unit circle with margin
       
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        x, y = y, x
        
        self.points = torch.tensor(np.column_stack((x, y)), dtype=torch.float32, device=self.device)

        grid_x, grid_y = torch.meshgrid(torch.linspace(-1, 1, grid_size, device=self.device),
                                        torch.linspace(-1, 1, grid_size, device=self.device), 
                                        indexing='ij')
        
        self.target_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        self.mask_indices = (self.target_points[:, 0]**2 + self.target_points[:, 1]**2) > 1.0

        # Squared distances from each grid point to each sensor
        dists_sq = torch.cdist(self.target_points, self.points).pow(2)

        # t-SNE-style binary search: find per-grid-point sigma so that
        # the Gaussian kernel over sensors has the desired perplexity
        weights = self._binsearch_perplexity(dists_sq, perplexity)

        weights[self.mask_indices, :] = 0.0
        self.register_buffer('weights', weights)

    @staticmethod
    def _binsearch_perplexity(dists_sq, target_perplexity, tol=1e-5, max_iter=50):
        """Binary search for per-row sigma that yields target perplexity.
        Scikit-learn's t-SNE uses a similar approach to find the bandwidth of the Gaussian kernel for each data point, such that the perplexity of the distribution over neighbors matches a specified value. This function implements that binary search.

        Args:
            dists_sq: (N, K) squared distances from N grid points to K sensors.
            target_perplexity: desired perplexity (effective number of neighbours).
            tol: convergence tolerance on log-perplexity.
            max_iter: maximum binary-search iterations.

        Returns:
            weights: (N, K) row-normalised Gaussian weights with adaptive sigma.
        """
        N, K = dists_sq.shape
        target_entropy = np.log(target_perplexity)  # ln-based

        # Precision beta = 1 / (2 * sigma^2); search in log-space
        beta = torch.ones(N, 1, device=dists_sq.device)
        beta_min = torch.full((N, 1), -float('inf'), device=dists_sq.device)
        beta_max = torch.full((N, 1), float('inf'), device=dists_sq.device)

        for _iter in range(max_iter):
            # Compute Gaussian kernel with current beta (precision)
            W = torch.exp(-dists_sq * beta)           # (N, K)
            sum_W = W.sum(dim=1, keepdim=True).clamp(min=1e-12)
            P = W / sum_W                              # row-normalised probs

            # Shannon entropy H = -sum(p * ln(p)), with 0*ln(0)=0
            log_P = torch.log(P.clamp(min=1e-12))
            H = -(P * log_P).sum(dim=1, keepdim=True)  # (N, 1)

            # Check convergence
            H_diff = H - target_entropy
            converged = H_diff.abs() < tol
            if converged.all():
                break

            # Binary-search update
            needs_increase = (H_diff > 0).squeeze(1)   # entropy too high → increase beta (shrink sigma)
            needs_decrease = (H_diff < 0).squeeze(1)    # entropy too low  → decrease beta (grow sigma)

            # Update bounds
            beta_min[needs_increase] = beta[needs_increase]
            beta_max[needs_decrease] = beta[needs_decrease]

            # Step
            finite_max = torch.isfinite(beta_max)
            finite_min = torch.isfinite(beta_min)

            # Both bounds finite → bisect
            both = (finite_max & finite_min).squeeze(1)
            beta[both] = (beta_min[both] + beta_max[both]) / 2.0

            # Only lower bound finite → double
            only_min = (finite_min & ~finite_max).squeeze(1) & needs_increase
            beta[only_min] = beta[only_min] * 2.0

            # Only upper bound finite → halve
            only_max = (~finite_min & finite_max).squeeze(1) & needs_decrease
            beta[only_max] = beta[only_max] / 2.0

        # Final normalised weights
        W = torch.exp(-dists_sq * beta)
        sum_W = W.sum(dim=1, keepdim=True).clamp(min=1e-12)
        weights = W / sum_W
        return weights

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

        '''
        with open(stats_path, 'r') as f:
            self.stats_lookup = json.load(f)
        '''

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

        #print(label_idx)
        file_path = os.path.join(self.samples_dir, fname)
        if self.preload:
            video = self.cache[fname]  # No clone needed - data is not modified in-place
        else:
            try:
                video = torch.load(file_path, weights_only=True, map_location='cpu').float()
            except Exception as e:
                print(f"Error loading {fname}: {e}")
                return torch.zeros(5, 32, 32, 32), torch.zeros(32, 32), 0
        
        #mean = video.mean(dim=(2,3,4), keepdim=True)
        #std = video.std(dim=(2,3,4), keepdim=True) + 1e-8


        #video = (video - mean) / (std)
        #print(video.shape)
        target_map = torch.zeros((self.grid_size, self.grid_size), dtype=torch.long)
        target_map[self.prototypes[label_idx] > 0.1] = label_idx + 1  # Background is 0
        if self.augmentations is not None and self.split == 'train':
            video1 = self.augmentations(video)
            video2 = self.augmentations(video)
            return video1, video2, target_map, label_idx, fname
        
        return video, target_map, label_idx, fname
    

class PKSampler(Sampler):
    def __init__(self, dataset, batch_size, n_classes=5, n_samples_per_class=None):
        # Access labels directly from dataset.samples: (filename, emotion_id, stats_key)
        self.labels = [s[1] for s in dataset.samples]
        print(f"PKSampler: {len(self.labels)} samples")
        
        self.labels = torch.tensor(self.labels).long()
        self.label_set = list(set(self.labels.numpy()))
        
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.label_set}
        
        for l in self.label_set:
            np.random.shuffle(self.label_to_indices[l])
            
        self.used_label_indices_count = {label: 0 for label in self.label_set}
        self.count = 0

        self.n_classes = min(n_classes, len(self.label_set))
        if self.n_classes < n_classes:
            print(f"PKSampler: Warning - only {self.label_set} classes are avalable!")
        
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