import torch
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

from data_process.data_loader import DatasetReader
from data_process.wavelet import WaveletModule
from src.snn_modeling.dataloader.dataset import TopoMapper

SELECTED_EMOTIONS = [3, 5, 8, 10, 15] 

def run_data_setup(config=None):
    print("Initializing Pipeline...")

    WINDOW_SIZE = config['data'].get('window_size', 256)
    STEP_SIZE = config['data'].get('step_size', 128)
    TARGET_STEPS = config['data'].get('num_timesteps', 32)
    SAMPLING_RATE = config['data'].get('sampling_rate', 256) 
    RAW_FOLDER = config['data']['raw_path']
    COORDS_PATH = config['data']['coords_path']
    OUTPUT_FOLDER = config['data']['dataset_path'] 

    if not os.path.exists(RAW_FOLDER):
        raise FileNotFoundError(f"Raw folder not found: {RAW_FOLDER}")
    if not os.path.exists(COORDS_PATH):
        raise FileNotFoundError(f"Coords file not found: {COORDS_PATH}")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    dataset_reader = DatasetReader(RAW_FOLDER, selected_emotions=SELECTED_EMOTIONS)
    wavelet = WaveletModule(window_size=WINDOW_SIZE, fs=SAMPLING_RATE, target_steps=TARGET_STEPS)
    coords = pd.read_csv(COORDS_PATH, sep='\t')
    topo = TopoMapper(coords, grid_size=64)
    
    all_videos = []
    all_labels = []
    emotion_map = {original: idx for idx, original in enumerate(SELECTED_EMOTIONS)}
    print(f"Processing {len(dataset_reader)} raw files...")

    for i in tqdm(range(len(dataset_reader))):
        try:
            raw_eeg, emotion_id = dataset_reader[i] 
            
            windows = wavelet.create_windows(raw_eeg, step_size=STEP_SIZE)
            
            if len(windows) == 0: continue
            
            for w_idx in range(len(windows)):
                window = windows[w_idx]
                w_mean = torch.mean(window)
                w_std = torch.std(window)
                limit = 6.0 * w_std # By Chebyshev inequality, covers > 97% data
                window = torch.clamp(window, w_mean - limit, w_mean + limit)
 
                raw_power = wavelet.wavelet(window)
   
                band_feats = wavelet.bandpowers_wavelet(raw_power)

                feats_32 = wavelet.resample_time(band_feats)

                feats_permuted = feats_32.permute(1, 2, 0)
                
                video = topo(feats_permuted)
                
                all_videos.append(video.numpy()) 
                all_labels.append(emotion_map[emotion_id])
                
        except Exception as e:
            print(f"⚠️ Error processing file {i}: {e}")
            continue
    
    print("Stacking data...")
    data_np = np.stack(all_videos)
    labels_np = np.array(all_labels) 
    
    print(f"Final Shape: {data_np.shape}")
    
    print("Normalizing to [0, 1]...")
    d_min = data_np.min()
    d_max = data_np.max()
    data_np = (data_np - d_min) / (d_max - d_min)
    
    print(f"Saving to {OUTPUT_FOLDER}...")
    np.save(os.path.join(OUTPUT_FOLDER, "data.npy"), data_np)
    np.save(os.path.join(OUTPUT_FOLDER, "labels.npy"), labels_np)

    print("Generation Complete.")
