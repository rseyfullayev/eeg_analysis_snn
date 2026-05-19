import torch
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import copy
from data_process.data_loader import DatasetReader
from data_process.wavelet import WaveletModule
from src.snn_modeling.dataloader.dataset import TopoMapper
import json
import torch.nn.functional as F

SELECTED_EMOTIONS = [0, 1, 2, 3, 4] 
NEUTRAL_EMOTION_ID = 3  # Index mapping: 0=Disgust, 1=Fear, 2=Sad, 3=Neutral, 4=Happy

def compute_alignment_matrix(covs_list, device):
    """
    Computes the EA alignment matrix R^{-1/2} and R^{1/2} from a list of trial covariances.
    """
    mean_cov = torch.stack(covs_list).mean(dim=0)
    C = mean_cov.shape[0]
    
    # Add small epsilon for numerical stability
    mean_cov = mean_cov + torch.eye(C, device=device) * 1e-4
    
    # Compute R^{-1/2} and R^{1/2} using Eigen Decomposition
    L, Q = torch.linalg.eigh(mean_cov)
    L_inv_sqrt = torch.diag(1.0 / torch.sqrt(L.clamp(min=1e-6)))
    R_inv_sqrt = torch.matmul(torch.matmul(Q, L_inv_sqrt), Q.t())
    
    L_sqrt = torch.diag(torch.sqrt(L.clamp(min=1e-6)))
    R_sqrt = torch.matmul(torch.matmul(Q, L_sqrt), Q.t())
    
    return R_inv_sqrt, R_sqrt

def apply_alignment(x, R_inv_sqrt):
    """
    Applies the pre-computed alignment matrix to the raw signal.
    """
    is_batched = (x.dim() == 3)
    if not is_batched:
        x = x.unsqueeze(0)
        
    # Center the data
    x_mean = x.mean(dim=-1, keepdim=True)
    x_centered = x - x_mean
    
    R_inv_sqrt_batched = R_inv_sqrt.unsqueeze(0).expand(x.shape[0], -1, -1)
    x_aligned = torch.bmm(R_inv_sqrt_batched, x_centered)
    
    if not is_batched:
        x_aligned = x_aligned.squeeze(0)
        
    return x_aligned

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

    
    return subject_id, session_id

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
    
    # --- EA PASS 1: COMPUTE SESSION COVARIANCES ---
    print(f"Phase 1: Computing Session-Level Euclidean Alignment Matrices across {len(dataset_reader)} files...")
    subject_covs = defaultdict(list)
    
    for raw_eeg, emotion_id, orig_filename in tqdm(dataset_reader.iterate_file_based(), total=len(dataset_reader), desc="EA Pass 1"):
        subject_id, session_id = parse_metadata(orig_filename)
        session_key = f"{subject_id}_{session_id}"
        
        raw_eeg = raw_eeg.to(device)
        
        # Calculate single trial covariance
        x_mean = raw_eeg.mean(dim=-1, keepdim=True)
        x_centered = raw_eeg - x_mean
        
        if x_centered.dim() == 3:
            x_centered = x_centered.squeeze(0) # (C, T)
            
        C, T = x_centered.shape
        cov = torch.mm(x_centered, x_centered.t()) / max(T - 1, 1)
        subject_covs[session_key].append(cov)

    print("Phase 1 Complete: Resolving Alignment Matrices...")
    subject_alignment_matrices = {}
    subject_r_sqrt_matrices = {}
    for subj, covs in subject_covs.items():
        r_inv, r_sqrt = compute_alignment_matrix(covs, device)
        subject_alignment_matrices[subj] = r_inv
        subject_r_sqrt_matrices[subj] = r_sqrt
        
    torch.save({
        'R_inv_sqrt': subject_alignment_matrices,
        'R_sqrt': subject_r_sqrt_matrices
    }, os.path.join(OUTPUT_FOLDER, "R_matrices.pt"))
    print(f"Saved R matrices to {os.path.join(OUTPUT_FOLDER, 'R_matrices.pt')}")

    # --- ERD Phase 1.5: Compute per-subject Neutral CWT baseline (C_k) ---
    # For each subject, compute the mean CWT magnitude spectrum across all their
    # Neutral trials. This captures the stationary IAF signature that will be
    # subtracted from every active emotion window.
    print(f"Phase 1.5: Computing per-subject Neutral CWT baselines (ERD)...")
    subject_neutral_accum = {}   # subject_id -> running sum of CWT magnitude
    subject_neutral_count = {}   # subject_id -> number of accumulated samples

    for raw_eeg, emotion_id, orig_filename in tqdm(
            dataset_reader.iterate_file_based(), total=len(dataset_reader), desc="ERD Neutral Scan"):
        if emotion_id != NEUTRAL_EMOTION_ID:
            continue

        subject_id, session_id = parse_metadata(orig_filename)
        session_key = f"{subject_id}_{session_id}"

        raw_eeg = raw_eeg.to(device)
        if raw_eeg.dim() == 2 and raw_eeg.shape[1] < WINDOW_SIZE:
            continue
        elif raw_eeg.dim() == 3 and raw_eeg.shape[2] < WINDOW_SIZE:
            continue

        with torch.no_grad():
            # Apply the same EA alignment as the main pipeline
            R_inv_sqrt = subject_alignment_matrices[session_key]
            raw_eeg = apply_alignment(raw_eeg, R_inv_sqrt)

            # CWT + log1p (same as Phase 2 below)
            feats = wavelet(raw_eeg)
            feats = torch.log1p(feats)
            # Robust scaling (same IQR normalization)
            median = feats.median(dim=-1, keepdim=True).values
            q1 = torch.quantile(feats, 0.25, dim=-1, keepdim=True)
            q3 = torch.quantile(feats, 0.75, dim=-1, keepdim=True)
            iqr = q3 - q1 + 1e-6
            feats = (feats - median) / iqr
            feats = torch.clamp(feats, -4.0, 4.0)

            # Mean across time dimension -> subject-level neutral spectrum
            # feats shape: [1, C, Bands, T] -> mean over T -> [1, C, Bands]
            neutral_mean = feats.mean(dim=-1)  # [1, C, Bands]

        if subject_id not in subject_neutral_accum:
            subject_neutral_accum[subject_id] = neutral_mean.clone()
            subject_neutral_count[subject_id] = 1
        else:
            subject_neutral_accum[subject_id] += neutral_mean
            subject_neutral_count[subject_id] += 1

    # Finalize: C_k = mean neutral CWT per subject
    subject_neutral_baselines = {}
    for subj_id in subject_neutral_accum:
        C_k = subject_neutral_accum[subj_id] / subject_neutral_count[subj_id]  # [1, C, Bands]
        subject_neutral_baselines[subj_id] = C_k
        print(f"  Subject {subj_id}: Neutral baseline from {subject_neutral_count[subj_id]} trials")

    if not subject_neutral_baselines:
        print("  WARNING: No neutral baselines computed! ERD subtraction will be skipped.")
    else:
        torch.save(subject_neutral_baselines, os.path.join(OUTPUT_FOLDER, "C_baselines.pt"))
        print(f"Saved C baselines to {os.path.join(OUTPUT_FOLDER, 'C_baselines.pt')}")

    # --- STATISTICS COLLECTOR ---

    registry = []
    sample_global_id = 0
    bag_id = 0
    class_counts = {i: 0 for i in range(len(SELECTED_EMOTIONS))}
    total_collected = 0
    GPU_BATCH_SIZE = 16 
    
    print("Phase 2: Generating Dataset (with ERD subtraction)...")    
    for raw_eeg, emotion_id, orig_filename in tqdm(dataset_reader.iterate_file_based(), total=len(dataset_reader)): 
        bag_id += 1
        subject_id, session_id = parse_metadata(orig_filename)
        session_key = f"{subject_id}_{session_id}"

        if use_sampling_limit and total_collected >= TOTAL_TARGET: break

        raw_eeg = raw_eeg.to(device)
        if raw_eeg.dim() == 2 and raw_eeg.shape[1] < WINDOW_SIZE: continue
        elif raw_eeg.dim() == 3 and raw_eeg.shape[2] < WINDOW_SIZE: continue

        with torch.no_grad():
            # Apply pre-computed Session-Level Euclidean Alignment
            R_inv_sqrt = subject_alignment_matrices[session_key]
            raw_eeg = apply_alignment(raw_eeg, R_inv_sqrt)
            
            feats = wavelet(raw_eeg)
            feats = torch.log1p(feats)
            median = feats.median(dim=-1, keepdim=True).values
            q1 = torch.quantile(feats, 0.25, dim=-1, keepdim=True)
            q3 = torch.quantile(feats, 0.75, dim=-1, keepdim=True)
            iqr = q3 - q1 + 1e-6

            feats = (feats - median) / iqr
            feats = torch.clamp(feats, -4.0, 4.0) # 1 iqr is typically ~1.5 std, so this is roughly 6 stds from the median, which should be safe for outliers

            # --- ERD: Subtract neutral baseline from active emotion windows ---
            # X_pure_emotion = X_active - C_k (broadcast over time dimension)
            if subject_id in subject_neutral_baselines:
                C_k = subject_neutral_baselines[subject_id]  # [1, C, Bands]
                # C_k has no time dim; feats is [1, C, Bands, T] -> broadcast subtract
                feats = feats - C_k.unsqueeze(-1)

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
                    fname = f"{subject_id}_{bag_id}_s{sample_global_id}.pt"
                    save_path = os.path.join(OUTPUT_FOLDER, fname)
                    
                    torch.save(video_batch[k].clone(), save_path)
                    
                    # bag_id -> Acts as our Bag ID (The unique Movie Clip)
                    registry.append(f"{fname},{bag_id},{emotion_id}")
                    sample_global_id += 1
                
            torch.cuda.empty_cache()

    with open(os.path.join(OUTPUT_FOLDER, "index.csv"), 'w') as f:

        f.write("filename,bag_id,emotion_id\n")
        for line in registry:
            f.write(f"{line}\n")
            
    print("Data Setup Complete.")