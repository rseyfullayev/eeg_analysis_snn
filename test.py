
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.snn_modeling.utils.loss import FullHybridLoss, TopKClassificationLoss, ContrastiveLoss
from src.snn_modeling.dataloader.dataset import SWEEPDataset, PKSampler
import re
import os
import gc
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import snntorch as snn
import segmentation_models_pytorch as smp
import wandb
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import numpy as np

from tqdm import tqdm
from src.snn_modeling.utils.utils import initialize_network
from src.snn_modeling.utils.augmentations import VideoTemporalMasking, GaussianNoise, FrequencyDropout, SignalJitter, VideoRandomErasing
from src.snn_modeling.layers.neurons import ALIF
from src.snn_modeling.models.unet import SpikingResNetClassifier
from src.snn_modeling.models.encoders import SpikingResNet18Encoder
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns


# Ignore the specific sklearn warning about missing classes
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
warnings.filterwarnings("ignore", message="A single label was found in 'y_true' and 'y_pred'. For the confusion matrix to have the correct shape, use the 'labels' parameter to pass all known labels.")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def test(config, loso, subj, device, model):

    masks = torch.load(os.path.join(config['data']['dataset_path'],'masks.pt')).to(device)

    val_set = SWEEPDataset(
    config, 
    split='val',
    #experiment=True,
    loso=loso,
    subj=subj,
    prototypes=masks
    )

    val_loader = DataLoader(val_set, 
                            batch_size=config['training']['batch_size'], 
                            shuffle=False, 
                            pin_memory=True)

    embs = []
    labels = []

    with torch.no_grad():
        for (vid, _, lbl, _) in val_loader:

            vid = vid.permute(1,0,2,3,4).to(device)

            emb, _ = model.encoder(vid)
            #print(emb.shape)
            emb = emb.mean([0,3,4])
            embs.append(emb)
            labels.append(lbl)

        emb = torch.cat(embs, dim=0).cpu().numpy()
        labels = torch.cat(labels, dim=0).cpu().numpy()

    print(emb.shape, labels.shape)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    emb_2d = reducer.fit_transform(emb)

    plt.figure(figsize=(10,8))
    palette = sns.color_palette("husl", 5)
    sns.scatterplot(
        x=emb_2d[:,0],
        y=emb_2d[:,1],
        hue=labels,
        palette=palette,
        s=15,
        alpha=.7
    )

    plt.title('UMAP Projection')
    plt.legend(title='Emotion', bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.savefig('evidence/test.png')

    

    