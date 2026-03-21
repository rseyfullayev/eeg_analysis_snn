import torch
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from data_process.data_loader import DatasetReader
from data_process.wavelet import WaveletModule
from src.snn_modeling.dataloader.dataset import TopoMapper
import json
import torch.nn.functional as F

SELECTED_EMOTIONS = [0, 1, 2, 3, 4] 

def parse_metadata(filename):
    """
    Format: SubjectID_SessionID_Date.cnt (e.g., "1_1_20180804.cnt")
    """

    clean_name = os.path.splitext(filename)[0]
    
    parts = clean_name.split('_')
    

    if len(parts) >= 2:
        subject_id = parts[0]  
        session_id = parts[1]  
    else:
        subject_id = parts[0]
        session_id = "unknown"

    
    trial_unique_id = clean_name 
    
    return subject_id, trial_unique_id

def run_data_setup(config=None):
    print("Initializing Pipeline...")

     # [Standard Config Setup - Same as before]
    WINDOW_SIZE = config.data.get('window_size', 256)
    STEP_SIZE = config.data.get('step_size', 128)
    TARGET_STEPS = config.data.get('num_timesteps', 32)
    SAMPLING_RATE = config.data.get('sampling_rate', 256) 
    RAW_FOLDER = config.data.raw_path
    COORDS_PATH = config.data.coords_path
    OUTPUT_FOLDER = config.data.dataset_path 
    TOTAL_TARGET = config.data.get('num_samples', None)

    # [Setup Limit Logic - Same as before]
    if TOTAL_TARGET:
        SAMPLES_PER_CLASS = TOTAL_TARGET // len(SELECTED_EMOTIONS)
        use_sampling_limit = True
    else:
        SAMPLES_PER_CLASS = float('inf')
        use_sampling_limit = False
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset_reader = DatasetReader(RAW_FOLDER)
    wavelet = WaveletModule(fs=SAMPLING_RATE, target_steps=TARGET_STEPS, device=device)
    coords = pd.read_csv(COORDS_PATH, sep=',')
    
    # Allow perplexity to be dynamically configured
    perplexity_val = config.data.get('perplexity', 5.0)
    topo = TopoMapper(coords, grid_size=config.data.get('grid_size', 32), perplexity=perplexity_val, device=device)

    emotion_map = {original: idx for idx, original in enumerate(SELECTED_EMOTIONS)}
    
    # --- STATISTICS COLLECTOR ---

    registry = []
    sample_global_id = 0
    class_counts = {i: 0 for i in range(len(SELECTED_EMOTIONS))}
    total_collected = 0
    GPU_BATCH_SIZE = 16 
    
    print(f"Processing {len(dataset_reader)} raw files...")
    
    for raw_eeg, emotion_id, orig_filename in tqdm(dataset_reader.iterate_file_based(), total=len(dataset_reader)): 
        
        subject_id, trial_id = parse_metadata(orig_filename)

        if use_sampling_limit and total_collected >= TOTAL_TARGET: break

        raw_eeg = raw_eeg.to(device)
        if raw_eeg.shape[1] < WINDOW_SIZE: continue

        with torch.no_grad():
            feats = wavelet(raw_eeg)
            feats = torch.log1p(feats)
            median = feats.median(dim=-1, keepdim=True).values
            q1 = torch.quantile(feats, 0.25, dim=-1, keepdim=True)
            q3 = torch.quantile(feats, 0.75, dim=-1, keepdim=True)
            iqr = q3 - q1 + 1e-6

            feats = (feats - median) / iqr
            feats = torch.clamp(feats, -4.0, 4.0) # 1 iqr is typically ~1.5 std, so this is roughly 6 stds from the median, which should be safe for outliers

            windows = feats.unfold(dimension=-1, size=WINDOW_SIZE, step=STEP_SIZE)

            windows = windows.squeeze(0).permute(2, 0, 1, 3)
            
            W, C, B, WS = windows.shape

            win_flat = windows.reshape(W, C * B, WS)
            win_flat = F.interpolate(win_flat, 
                                     size=TARGET_STEPS, 
                                     mode='linear', 
                                     align_corners=False)
            
            full_batch_tensor = win_flat.reshape(W, C, B, TARGET_STEPS)
            full_batch_tensor = full_batch_tensor.permute(0, 3, 2, 1)

        if use_sampling_limit:
            if emotion_id not in emotion_map: continue
            label_idx = emotion_map[emotion_id]
            if class_counts[label_idx] >= SAMPLES_PER_CLASS: continue
            needed = SAMPLES_PER_CLASS - class_counts[label_idx]
            full_batch_tensor = full_batch_tensor[:needed]
            class_counts[label_idx] += full_batch_tensor.shape[0]
            total_collected += full_batch_tensor.shape[0]

        if full_batch_tensor.size(0) == 0: continue

        for i in range(0, full_batch_tensor.size(0), GPU_BATCH_SIZE):
            batch_tensor = full_batch_tensor[i : i + GPU_BATCH_SIZE]
            
            with torch.no_grad():

    

                # Topo & Save
                video_batch = topo(batch_tensor)
                video_batch = video_batch.cpu()

                for k in range(video_batch.shape[0]):
                    # Unique filename for the window
                    fname = f"{subject_id}_{trial_id}_s{sample_global_id}.pt"
                    save_path = os.path.join(OUTPUT_FOLDER, fname)
                    
                    torch.save(video_batch[k].clone(), save_path)
                    
                    # trial_id -> Acts as our Bag ID (The unique Movie Clip)
                    registry.append(f"{fname},{trial_id},{emotion_id}")
                    sample_global_id += 1
                
            torch.cuda.empty_cache()

    with open(os.path.join(OUTPUT_FOLDER, "index.csv"), 'w') as f:

        f.write("filename,bag_id,emotion_id\n")
        for line in registry:
            f.write(f"{line}\n")
            
    print("Data Setup Complete.")