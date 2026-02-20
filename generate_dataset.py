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

    stats_key = f"{subject_id}_{session_id}"
    
    trial_unique_id = clean_name 
    
    return stats_key, subject_id, trial_unique_id

def run_data_setup(config=None):
    print("Initializing Pipeline...")

     # [Standard Config Setup - Same as before]
    WINDOW_SIZE = config['data'].get('window_size', 256)
    STEP_SIZE = config['data'].get('step_size', 128)
    TARGET_STEPS = config['data'].get('num_timesteps', 32)
    SAMPLING_RATE = config['data'].get('sampling_rate', 256) 
    RAW_FOLDER = config['data']['raw_path']
    COORDS_PATH = config['data']['coords_path']
    OUTPUT_FOLDER = config['data']['dataset_path'] 
    TOTAL_TARGET = config['data'].get('num_samples', None)

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
    topo = TopoMapper(coords, grid_size=config['data'].get('grid_size', 32), device=device)

    emotion_map = {original: idx for idx, original in enumerate(SELECTED_EMOTIONS)}
    
    # --- STATISTICS COLLECTOR ---

    stats_reservoir = defaultdict(list)
    RESERVOIR_LIMIT = 20000 

    registry = []
    sample_global_id = 0
    class_counts = {i: 0 for i in range(len(SELECTED_EMOTIONS))}
    total_collected = 0
    GPU_BATCH_SIZE = 16 
    
    print(f"Processing {len(dataset_reader)} raw files...")
    
    for raw_eeg, emotion_id, orig_filename in tqdm(dataset_reader.iterate_file_based(), total=len(dataset_reader)): 
        
        stats_key, subject_id, trial_id = parse_metadata(orig_filename)

        if use_sampling_limit and total_collected >= TOTAL_TARGET: break

        raw_eeg = raw_eeg.to(device)
        if raw_eeg.shape[1] < WINDOW_SIZE: continue

        feats = wavelet(raw_eeg)
        feats = torch.log1p(feats)

        windows = feats.unfold(1, WINDOW_SIZE, STEP_SIZE)

        windows = windows.permute(1, 0, 2)

        windows = F.interpolate(
            windows,
            size=TARGET_STEPS,
            mode='linear',
            align_corners=False
        )

        full_batch_tensor = windows

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

                # Collect Stats (Per Session/Subject)
                if len(stats_reservoir[stats_key]) < RESERVOIR_LIMIT:
                    flat_data = feats.flatten().cpu().numpy()
                    indices = np.random.randint(0, len(flat_data), size=min(100, len(flat_data)))
                    stats_reservoir[stats_key].extend(flat_data[indices])

                # Topo & Save
                video_batch = topo(full_batch_tensor)
                video_batch = torch.clamp(video_batch, min=0.0, max=50.0)  # 10^50 is impossible thus clipping

                video_batch = video_batch.cpu()
                for k in range(video_batch.shape[0]):
                    # Unique filename for the window
                    fname = f"{subject_id}_{trial_id}_s{sample_global_id}.pt"
                    save_path = os.path.join(OUTPUT_FOLDER, fname)
                    
                    torch.save(video_batch[k].clone(), save_path)
                    
                    # stats_key -> Used to look up Mean/Std later
                    # trial_id -> Used for GroupShuffleSplit (The Video)
                    registry.append(f"{fname},{stats_key},{trial_id},{emotion_id}")
                    sample_global_id += 1
                
            torch.cuda.empty_cache()

    # --- SAVE ---
    print("Computing Stats...")
    final_stats = {}
    for key, values in stats_reservoir.items():
        arr = np.array(values)
        final_stats[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p98": float(np.percentile(arr, 98))
        }

    with open(os.path.join(OUTPUT_FOLDER, "stats.json"), 'w') as f:
        json.dump(final_stats, f, indent=4)

    with open(os.path.join(OUTPUT_FOLDER, "index.csv"), 'w') as f:

        f.write("filename,stats_key,group_id,emotion_id\n")
        for line in registry:
            f.write(f"{line}\n")
            
    print("Data Setup Complete.")