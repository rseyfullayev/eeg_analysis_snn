import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader



class DatasetReader(Dataset):
    def __init__(self, folder_path, selected_emotions=None):
        self.folder_path = folder_path
        self.selected = set(selected_emotions) if selected_emotions else None
        self.files = []

        for fname in os.listdir(folder_path):
            if not fname.endswith(".txt"):
                continue


            parts = fname.replace(".txt", "").split("_")
            emotion_id = int([p for p in parts if p.isdigit()][0])

            if self.selected is None or emotion_id in self.selected:
                self.files.append(os.path.join(folder_path, fname))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        full_path = self.files[idx]

        df = pd.read_csv(full_path, sep="\t", header=None)
        eeg = torch.tensor(df.values, dtype=torch.float32).T  # (channels, samples)


        fname = os.path.basename(full_path)
        emotion_id = int([p for p in fname.split("_") if p.split(".")[0].isdigit()][0])

        return eeg, emotion_id
