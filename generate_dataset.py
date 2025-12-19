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
    
    if not os.path.exists(RAW_FOLDER):
        raise FileNotFoundError(f"Raw folder not found: {RAW_FOLDER}")
    if not os.path.exists(COORDS_PATH):
        raise FileNotFoundError(f"Coords file not found: {COORDS_PATH}")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_reader = DatasetReader(RAW_FOLDER)
    wavelet = WaveletModule(fs=SAMPLING_RATE, target_steps=TARGET_STEPS, device=device)
    coords = pd.read_csv(COORDS_PATH, sep=',')
    topo = TopoMapper(coords, grid_size=config['data'].get('grid_size', 64))

    emotion_map = {original: idx for idx, original in enumerate(SELECTED_EMOTIONS)}
    print(f"Processing {len(dataset_reader)} raw files...")
    class_counts = {i: 0 for i in range(len(SELECTED_EMOTIONS))}
    total_collected = 0
    indices = np.arange(len(dataset_reader))
    np.random.shuffle(indices)

    RESERVOIR_SIZE = 10_000_000 # Glivenko-Cantelli theorem, large sample for percentile estimation
    p98_reservoir = torch.zeros(RESERVOIR_SIZE, dtype=torch.float32)
    res_idx = 0
    res_full = False

    sample_id = 0
    registry = []
    GPU_BATCH_SIZE = 16 
    for raw_eeg, emotion_id, orig_filename  in tqdm(dataset_reader.iterate_file_based(), total=len(dataset_reader), desc="Files Processed"): 
        if use_sampling_limit and total_collected >= TOTAL_TARGET:
            print("\n Total sample target reached!")
            break

        raw_eeg = raw_eeg.to(device)
        if raw_eeg.shape[1] < WINDOW_SIZE:
            continue

        windows = raw_eeg.unfold(1, WINDOW_SIZE, STEP_SIZE)
        full_batch_tensor  = windows.permute(1, 0, 2)
        

        if use_sampling_limit:
            if emotion_id not in emotion_map: continue
            label_idx = emotion_map[emotion_id]
            if class_counts[label_idx] >= SAMPLES_PER_CLASS:
                continue
            needed = SAMPLES_PER_CLASS - class_counts[label_idx]
            full_batch_tensor  = full_batch_tensor [:needed]
            current_batch_size = full_batch_tensor .shape[0]
            class_counts[label_idx] += current_batch_size
            total_collected += current_batch_size
        #print(batch_tensor.shape)    
        if full_batch_tensor.size(0) == 0: continue
        for i in range(0, full_batch_tensor.size(0), GPU_BATCH_SIZE):
            batch_tensor = full_batch_tensor[i : i + GPU_BATCH_SIZE]
            with torch.no_grad():
                feats = wavelet(batch_tensor)
                video_batch = topo(feats)
                video_batch = torch.clamp(video_batch, min=0.0)
                if video_batch.ndim != 5:
                    print(f"❌ CRITICAL ERROR: Output shape is {video_batch.shape}. Expected 5D (B, 5, 32, 32, 32).")
                    print(f"   Wavelet Output was: {feats.shape}")
                    break
                if not res_full:
                    flat_feats = feats.flatten()
                    active_pixels = flat_feats[flat_feats > 1e-9]
                    if active_pixels.numel() > 0:
                        num_to_take = min(1000, active_pixels.numel())
                        indices = torch.randint(0, active_pixels.numel(), (num_to_take,), device=device)
                        sampled_pixels = active_pixels[indices]
                        end_ptr = res_idx + num_to_take
                        if end_ptr < RESERVOIR_SIZE:
                            p98_reservoir[res_idx:end_ptr] = sampled_pixels.cpu()
                            res_idx = end_ptr
                        else:
                            remaining = RESERVOIR_SIZE - res_idx
                            p98_reservoir[res_idx:] = sampled_pixels[:remaining].cpu()
                            res_full = True
                            print("Reservoir full for percentile estimation.")

                video_batch = video_batch.cpu()
                for k in range(video_batch.shape[0]):
                    fname = f"{orig_filename}_s{sample_id}_lbl{emotion_id}.pt"
                    save_path = os.path.join(OUTPUT_FOLDER, fname)
                    torch.save(video_batch[k].clone(), save_path)
                    registry.append(f"{fname},{emotion_id}")
                    sample_id += 1
                torch.cuda.empty_cache()
            
    index_path = os.path.join(OUTPUT_FOLDER, "index.csv")
    with open(index_path, 'w') as f:
        f.write("filename,emotion_id\n")
        for line in registry:
            f.write(f"{line}\n")
    final_data = p98_reservoir[:res_idx].numpy()
    p98_value = np.percentile(final_data, 98)
    print(f"Estimated 98th percentile value across dataset: {p98_value:.4f}")
    print("Data processing complete.")
