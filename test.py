import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.snn_modeling.dataloader.dataset import SWEEPDataset
import os
import numpy as np
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import silhouette_score

import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def test(config, loso, subj, device, model):
    """Extract encoder embeddings from the validation set and visualize with UMAP.

    Args:
        config: OmegaConf config object.
        loso: Leave-one-subject-out ID (int or None).
        subj: Subject ID (int or None).
        device: torch device.
        model: A SpikingResNetClassifier (or anything with an .encoder attribute).
    """

    model.eval()

    masks = torch.load(os.path.join(config.data.dataset_path, 'masks.pt'), weights_only=True).to(device)

    val_set = SWEEPDataset(
        config,
        split='val',
        loso=loso,
        subj=subj,
        prototypes=masks
    )

    num_workers = config.data.get('num_workers', 0)
    prefetch = config.data.get('prefetch_factor', 2) if num_workers > 0 else None
    persist = num_workers > 0

    val_loader = DataLoader(
        val_set,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch,
        persistent_workers=persist,
        pin_memory=True
    )

    embs = []
    labels_list = []

    with torch.no_grad():
        for batch in val_loader:
            # Handle both augmented (6-item) and non-augmented (4-item) returns
            vid = batch[0]
            lbl = batch[2] if len(batch) >= 4 else batch[1]

            # Handle bag-level 6D inputs: [B, K, T, C, H, W]
            K_bag = None
            if vid.dim() == 6:
                B, K_bag, T, C, H, W = vid.shape
                vid = vid.view(B * K_bag, T, C, H, W)
            else:
                B, T, C, H, W = vid.shape

            # SNN requires Time as dimension 0: [T, Batch, C, H, W]
            vid = vid.permute(1, 0, 2, 3, 4).to(device)

            features, _ = model.encoder(vid)

            # Spatiotemporal GAP: (T, B, C, H, W) -> (B, C)
            if features.dim() == 5:
                emb = features.mean(dim=[0, 3, 4])
            else:
                emb = features.mean(dim=[-2, -1])

            # If bagged, average across K windows per bag
            if K_bag is not None and emb.shape[0] == B * K_bag:
                emb = emb.view(B, K_bag, -1).mean(dim=1)

            # L2-normalize embeddings for cosine UMAP
            emb = F.normalize(emb, dim=1, eps=1e-6)

            embs.append(emb.cpu())
            labels_list.append(lbl)

    emb_all = torch.cat(embs, dim=0).numpy()
    labels_all = torch.cat(labels_list, dim=0).numpy()

    print(f"Embeddings: {emb_all.shape} | Labels: {labels_all.shape}")

    # --- UMAP Projection ---
    reducer = umap.UMAP(n_neighbors=50, min_dist=0.01, metric='cosine', random_state=42)
    emb_2d = reducer.fit_transform(emb_all)

    # --- Clustering quality metric ---
    sil_score = silhouette_score(emb_all, labels_all, metric='cosine') if len(set(labels_all)) > 1 else 0.0
    print(f"Silhouette Score (cosine): {sil_score:.4f}")

    os.makedirs('evidence', exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    palette = sns.color_palette("husl", config.model.get('num_classes', 5))
    sns.scatterplot(
        x=emb_2d[:, 0],
        y=emb_2d[:, 1],
        hue=labels_all,
        palette=palette,
        s=15,
        alpha=0.7,
        ax=ax
    )

    id_label = f"LOSO {loso}" if loso else f"Subj {subj}"
    ax.set_title(f"UMAP — {id_label}  |  Silhouette: {sil_score:.3f}")
    ax.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()

    save_path = f"evidence/umap_loso{loso}.png" if loso else f"evidence/umap_subj{subj}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"UMAP saved to {save_path}")

    # --- W&B Logging ---
    wandb.log({
        f"Eval/UMAP_{id_label}": wandb.Image(save_path, caption=f"UMAP {id_label}"),
        f"Eval/Silhouette_{id_label}": sil_score,
        f"Eval/Num_Samples": len(labels_all),
    })
    print(f"Logged to W&B under Eval/ prefix.")