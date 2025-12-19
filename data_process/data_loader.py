import os
import numpy as np
import pandas as pd
import mne
import torch
from torch.utils.data import Dataset
import ast

class DatasetReader(Dataset):
    def __init__(self, root_dir):

        self.root_dir = root_dir
        self.eeg_dir = os.path.join(root_dir, 'EEG_raw')
   
        self.label_file = os.path.join(root_dir, 'emotion_label_and_stimuli_order.xlsx')
        self.time_file = os.path.join(root_dir, 'trial_start_end_timestamp.txt')
        
        self.labels = self._parse_labels()
        self.timestamps = self._parse_timestamps()
        
        self.file_index = []
        self._scan_files()
        
        print(f"Found {len(self.file_index)} total movie trials.")

    def iterate_file_based(self):
        """
        Generator that yields (raw_eeg_tensor, label) but optimizes I/O.
        It loads a CNT file ONCE, processes all 15 trials, then moves to the next.
        """
        # 1. Group indices by file path
        # Map: path -> list of {start, end, label}
        files_map = {}
        for item in self.file_index:
            path = item['path']
            if path not in files_map:
                files_map[path] = []
            files_map[path].append(item)

        # 2. Iterate through files
        print(f"Optimized Iterator: Processing {len(files_map)} unique files...")
        
        for path, trials in files_map.items():
            try:
                # LOAD ONCE
                mne.set_log_level('WARNING')
                raw = mne.io.read_raw_cnt(path, preload=True)
                useless_ch = ['M1', 'M2', 'VEO', 'HEO']
                raw.drop_channels(useless_ch, on_missing='ignore') 
                # Pick EEG
                raw.pick_types(eeg=True)
                if len(raw.ch_names) != 62:
                    print(f"⚠️ Channel Mismatch in: Has {len(raw.ch_names)} channels. Expected 62.")
                    print(f"   Channels found: {raw.ch_names}")
                    print(f"   File: {path}")  
                    if len(raw.ch_names) > 62:
                        raw.pick_channels(raw.ch_names[:62])
                    else:
                        continue
                    
                if raw.info['sfreq'] != 200:
                    raw.resample(200)
                base_filename = os.path.basename(path).replace('.cnt', '')

                for trial_meta in trials:
                    start_sample = int(trial_meta['t_start'] * 200)
                    end_sample = int(trial_meta['t_end'] * 200)
 
                    data, _ = raw[:, start_sample:end_sample]
                    max_val = np.max(np.abs(data))
                    if max_val > 0 and max_val < 1e-2: 
                        data = data * 1e6
                    eeg_tensor = torch.tensor(data, dtype=torch.float32)
                    yield eeg_tensor, int(trial_meta['label']), base_filename
                                    
            except Exception as e:
                print(f"Error reading file {path}: {e}")
                continue

    def _parse_labels(self):

        df = pd.read_excel(self.label_file, header=None, sheet_name="Primary")
        label_dict = {}
        for i in range(1,4): label_dict[i] = df.iloc[19+i-1, 2:17].dropna().astype(int).tolist()
        print(label_dict)
        return label_dict
            


    def _parse_timestamps(self):
        timestamp_dict = {}
        with open(self.time_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        i = 0
        while i < len(lines):
            try:
                session_line = lines[i]
                session_id = int(''.join(filter(str.isdigit, session_line)))
                
                starts = ast.literal_eval(lines[i+1][13:])
                ends = ast.literal_eval(lines[i+2][11:])
                timestamp_dict[session_id] = list(zip(starts, ends))
                i += 3
            except Exception as e:
                print(f"Error parsing timestamps at line {i}: {e}")
                i += 1
        return timestamp_dict


    def _scan_files(self):
        files = sorted([f for f in os.listdir(self.eeg_dir) if f.endswith('.cnt')])
        
        for f in files:
            try:
                parts = f.split('_')
                session_id = int(parts[1])
            except:
                continue

            if session_id not in self.labels:
                continue
            current_labels = self.labels[session_id]
            
            if session_id in self.timestamps:
                current_times = self.timestamps[session_id]
            else:
                current_times = self.timestamps.get(1, [])
                
            if not current_times:
                print(f"No timestamps for Session {session_id} in {f}")
                continue

            num_trials = min(len(current_labels), len(current_times))
            
            for trial_idx in range(num_trials): 
                self.file_index.append({
                    'path': os.path.join(self.eeg_dir, f),
                    'trial_idx': trial_idx,
                    'session_id': session_id,
                    'label': current_labels[trial_idx], 
                    't_start': current_times[trial_idx][0],
                    't_end': current_times[trial_idx][1]
                })

    def __len__(self):
        return len(self.file_index)

    def __getitem__(self, idx):
        meta = self.file_index[idx]
        
        try:
            mne.set_log_level('WARNING')
            raw = mne.io.read_raw_cnt(meta['path'], preload=True)
            useless_ch = ['M1', 'M2', 'VEO', 'HEO', 'CB1', 'CB2']
            raw.drop_channels(useless_ch, on_missing='ignore') 
            # Pick EEG
            raw.pick_types(eeg=True)
            if len(raw.ch_names) != 62:
                print(f"Channel Mismatch in: Has {len(raw.ch_names)} channels. Expected 62.")
                print(f"Channels found: {raw.ch_names}")
                if len(raw.ch_names) > 62:
                    raw.pick_channels(raw.ch_names[:62])
                else:
                    return torch.zeros(62, 200), -1
            raw.pick_types(eeg=True)
        
            if raw.info['sfreq'] != 200:
                raw.resample(200)
            
            start_sample = meta['t_start'] * 200
            end_sample = meta['t_end'] * 200

            data, _ = raw[:, start_sample:end_sample]
            max_val = np.max(np.abs(data))
            if max_val > 0 and max_val < 1e-2: 
                data = data * 1e6
            eeg_tensor = torch.tensor(data, dtype=torch.float32)

            if eeg_tensor.shape[0] != 62:
                eeg_tensor = eeg_tensor[:62, :]

            return eeg_tensor, int(meta['label'])
            
        except Exception as e:
            print(f"Error loading {meta['path']} Trial {meta['trial_idx']}: {e}")
            return torch.zeros(62, 200), -1