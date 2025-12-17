import torch
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import scipy.signal
from data_process.data_loader import DatasetReader
from data_process.wavelet import WaveletModule
from src.snn_modeling.dataloader.dataset import TopoMapper

SELECTED_EMOTIONS = [0, 1, 2, 3, 4] 

def run_data_setup(config=None):
    print("Initializing Pipeline...")

    WINDOW_SIZE = config['data'].get('window_size', 256)
    STEP_SIZE = config['data'].get('step_size', 128)
    TARGET_STEPS = config['data'].get('num_timesteps', 32)
    SAMPLING_RATE = config['data'].get('sampling_rate', 256) 
    RAW_FOLDER = config['data']['raw_path']
    COORDS_PATH = config['data']['coords_path']
    OUTPUT_FOLDER = config['data']['dataset_path'] 
    TOTAL_TARGET = config['data'].get('num_samples', None)
    if TOTAL_TARGET:
        print(f"Limited Sampling Mode: Target = {TOTAL_TARGET} total samples.")
        SAMPLES_PER_CLASS = TOTAL_TARGET // len(SELECTED_EMOTIONS)
        use_sampling_limit = True
    else:
        print("Full Dataset Mode: Processing all available files.")
        SAMPLES_PER_CLASS = float('inf') # Effectively no limit per class
        use_sampling_limit = False
    
    SAMPLES_PER_CLASS = TOTAL_TARGET // len(SELECTED_EMOTIONS)
    if not os.path.exists(RAW_FOLDER):
        raise FileNotFoundError(f"Raw folder not found: {RAW_FOLDER}")
    if not os.path.exists(COORDS_PATH):
        raise FileNotFoundError(f"Coords file not found: {COORDS_PATH}")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    dataset_reader = DatasetReader(RAW_FOLDER)
    wavelet = WaveletModule(window_size=WINDOW_SIZE, fs=SAMPLING_RATE, target_steps=TARGET_STEPS)
    coords = pd.read_csv(COORDS_PATH, sep=',')
    topo = TopoMapper(coords, grid_size=config['data'].get('grid_size', 64))
    
    all_videos = []
    all_labels = []
    emotion_map = {original: idx for idx, original in enumerate(SELECTED_EMOTIONS)}
    print(f"Processing {len(dataset_reader)} raw files...")
    class_counts = {i: 0 for i in range(len(SELECTED_EMOTIONS))}
    total_collected = 0
    indices = np.arange(len(dataset_reader))
    np.random.shuffle(indices)

    for raw_eeg, emotion_id in tqdm(dataset_reader.iterate_file_based(), total=len(dataset_reader), desc="Files Processed"):
        if use_sampling_limit and total_collected >= TOTAL_TARGET:
            print("\n Total sample target reached!")
            break

        try:
            if not use_sampling_limit:
                windows = wavelet.create_windows(raw_eeg, step_size=STEP_SIZE)
                if len(windows) == 0: continue
                    
                for window in windows:
                    window = scipy.signal.detrend(window, axis=-1)
                    w_mean = np.mean(window)
                    w_std = np.std(window)
                    window = np.clip(window, w_mean - 6*w_std, w_mean + 6*w_std)
                    window = torch.tensor(window, dtype=torch.float32)
                    
                    raw_power = wavelet.wavelet(window)
                    band_feats = wavelet.bandpowers_wavelet(raw_power)
                    feats_32 = wavelet.resample_time(band_feats)
                    feats_permuted = feats_32.permute(1, 2, 0)
                    video = topo(feats_permuted)
                    video = torch.clamp(video, min=0.0)
                    all_videos.append(video.cpu().numpy())
                    all_labels.append(emotion_map[emotion_id])
                continue

            if emotion_id not in emotion_map: continue
            label_idx = emotion_map[emotion_id]
            if class_counts[label_idx] >= SAMPLES_PER_CLASS:
                continue

            windows = wavelet.create_windows(raw_eeg, step_size=STEP_SIZE)
            
            if len(windows) == 0: continue

            needed = SAMPLES_PER_CLASS - class_counts[label_idx]
            windows = windows[:needed]
            
            for w_idx in range(len(windows)):
                window = windows[w_idx]
                window = scipy.signal.detrend(window, axis=-1)
                w_mean = np.mean(window)
                w_std = np.std(window)
                window = np.clip(window, w_mean - 6*w_std, w_mean + 6*w_std)
                window = torch.tensor(window, dtype=torch.float32)
 
                raw_power = wavelet.wavelet(window)
   
                band_feats = wavelet.bandpowers_wavelet(raw_power)

                feats_32 = wavelet.resample_time(band_feats)

                feats_permuted = feats_32.permute(1, 2, 0)
                
                video = topo(feats_permuted)
                video = torch.clamp(video, min=0.0)
                all_videos.append(video.cpu().numpy()) 
                all_labels.append(label_idx)

            count_added = len(windows)
            class_counts[label_idx] += count_added
            total_collected += count_added
                
        except Exception as e:
            print(f"Error processing file: {e}")
            continue
    
    print("Stacking data...")
    data_np = np.stack(all_videos)
    labels_np = np.array(all_labels) 

    all_pixels = data_np.flatten()
    brain_pixels  = all_pixels[all_pixels > 1e-12] 
    p98 = np.percentile(brain_pixels , 98)

    print(f"Final Shape: {data_np.shape}")
    print(f"98th Percentile Value (for clipping): {p98:.6f}")
    
    print(f"Saving to {OUTPUT_FOLDER}...")
    np.save(os.path.join(OUTPUT_FOLDER, "data.npy"), data_np)
    np.save(os.path.join(OUTPUT_FOLDER, "labels.npy"), labels_np)

    print("Generation Complete.")
