import pandas as pd
import torch
from torch.utils.data import Dataset
import os


class DatasetReader(Dataset):
    def __init__(self, folder_path, selected_emotions):
        self.folder_path = folder_path
        self.selected = set(selected_emotions)
        self.files = []

        for fname in os.listdir(folder_path):
            if fname.endswith(".txt"):
                emotion_id = int(fname.split("_")[1].split(".")[0])

                if emotion_id in self.selected:
                    self.files.append(os.path.join(folder_path, fname))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        full_path = self.files[idx]

        df = pd.read_csv(full_path, sep="\t", header=None)
        data_np = df.values

        eeg = torch.tensor(data_np, dtype=torch.float32).T

        fname = os.path.basename(full_path)
        emotion_id = int(fname.split("_")[1].split(".")[0])

        return eeg, emotion_id

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from wavelet import WaveletModule

    dataset = DatasetReader(
        folder_path="eeg_raw",
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    wavelet = WaveletModule()
    for eeg, emotion in loader:

        eeg = eeg.squeeze(0)
        print("eeg shape:", eeg.shape)
        print("emotion:", emotion.item())


        windows = wavelet.create_windows(eeg)
        print("windows:", windows.shape)

        coeffs = wavelet.wavelet(windows[0])
        print("coeffs:", coeffs.shape)

        bandpowers = wavelet.bandpowers_wavelet(coeffs)
        print("bandpowers:", bandpowers.shape)
        break
