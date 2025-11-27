from src.snn_modeling.dataloader.dataset import SWEEPDataset, TopoMapper
import pandas as pd

def setup_data(config):
    coords = pd.read_csv(config['data']['coords_path'], sep='\t')
    mapper = TopoMapper(coords)
    dataset = SWEEPDataset(config, mode='synthetic', transform=mapper)
    dataset.save_to_disk("data/processed_synthetic")