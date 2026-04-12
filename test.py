import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.snn_modeling.dataloader.dataset import SWEEPDataset
import os
import numpy as np
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import silhouette_score, balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics.pairwise import cosine_similarity

import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def _knn_accuracy(emb, labels, groups, k_values=(1, 3, 5, 10), n_folds=5):
    """Stratified Group k-fold k-NN accuracy to prevent leakage."""
    results = {}
    
    # Use StratifiedGroupKFold to prevent session/window leakage
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for k in k_values:
        preds_all, labels_all = [], []
        for train_idx, test_idx in sgkf.split(emb, labels, groups=groups):
            clf = KNeighborsClassifier(n_neighbors=k, metric='cosine')
            clf.fit(emb[train_idx], labels[train_idx])
            preds_all.append(clf.predict(emb[test_idx]))
            labels_all.append(labels[test_idx])

        preds_all = np.concatenate(preds_all)
        labels_all = np.concatenate(labels_all)
        bal_acc = balanced_accuracy_score(labels_all, preds_all)
        results[k] = bal_acc

    return results


def _linear_probe_accuracy(emb, labels, groups, n_folds=5):
    """Stratified Group k-fold logistic regression linear probe to prevent leakage."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    preds_all, labels_all = [], []

    for train_idx, test_idx in sgkf.split(emb, labels, groups=groups):
        clf = LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
            multi_class='multinomial',
            C=1.0
        )
        clf.fit(emb[train_idx], labels[train_idx])
        preds_all.append(clf.predict(emb[test_idx]))
        labels_all.append(labels[test_idx])

    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)
    return balanced_accuracy_score(labels_all, preds_all)


def _class_cosine_matrix(emb, labels, num_classes=5):
    """Compute intra-class (diagonal) and inter-class (off-diagonal) cosine similarity.
    
    Diagonal: mean pairwise cosine within each class (not trivial 1.0).
    Off-diagonal: cosine between class centroids.
    """
    centroids = []
    intra_sims = []

    for c in range(num_classes):
        mask = labels == c
        class_emb = emb[mask]
        if len(class_emb) < 2:
            centroids.append(class_emb[0] if len(class_emb) == 1 else np.zeros(emb.shape[1]))
            intra_sims.append(1.0)
            continue

        centroids.append(class_emb.mean(axis=0))
        # Intra-class: mean pairwise cosine (sample up to 500 for speed)
        if len(class_emb) > 500:
            idx = np.random.choice(len(class_emb), 500, replace=False)
            class_emb = class_emb[idx]
        pair_sim = cosine_similarity(class_emb)
        # Exclude self-similarity diagonal
        np.fill_diagonal(pair_sim, 0.0)
        n = pair_sim.shape[0]
        intra_sims.append(pair_sim.sum() / (n * (n - 1)))

    centroids = np.stack(centroids)
    inter_matrix = cosine_similarity(centroids)  # (C, C)

    # Replace diagonal with intra-class similarities
    for c in range(num_classes):
        inter_matrix[c, c] = intra_sims[c]

    return inter_matrix


def test(config, loso, subj, device, model):
    """Extract encoder embeddings from the validation set, compute metrics, and visualize.

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
    groups_list = []

    with torch.no_grad():
        for batch in val_loader:
            # Handle both augmented (6-item) and non-augmented (4-item) returns
            vid = batch[0]
            lbl = batch[2] if len(batch) >= 4 else batch[1]
            bag_id = batch[-1]  # The dataset ALWAYS returns bag_id as the last element

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
            groups_list.extend(bag_id)  # bag_id is a tuple/list of strings

    emb_all = torch.cat(embs, dim=0).numpy()
    labels_all = torch.cat(labels_list, dim=0).numpy()
    
    # Map string bag_ids to integer groups for CV
    unique_groups = {b_id: i for i, b_id in enumerate(set(groups_list))}
    groups_all = np.array([unique_groups[b_id] for b_id in groups_list])
    num_classes = config.model.get('num_classes', 5)
    id_label = f"LOSO {loso}" if loso else f"Subj {subj}"

    print(f"Embeddings: {emb_all.shape} | Labels: {labels_all.shape}")

    # =====================================================================
    # 1. Silhouette Score
    # =====================================================================
    sil_score = silhouette_score(emb_all, labels_all, metric='cosine') if len(set(labels_all)) > 1 else 0.0
    print(f"Silhouette Score (cosine): {sil_score:.4f}")

    # =====================================================================
    # 2. k-NN Accuracy (Grouped by bag_id to prevent leakage)
    # =====================================================================
    knn_results = _knn_accuracy(emb_all, labels_all, groups_all, k_values=(1, 3, 5, 10))
    print(f"\n--- k-NN Balanced Accuracy ({id_label}) ---")
    for k, acc in knn_results.items():
        print(f"  k={k:>2d}: {acc:.4f}")

    # =====================================================================
    # 3. Linear Probe (Logistic Regression) (Grouped by bag_id to prevent leakage)
    # =====================================================================
    lp_acc = _linear_probe_accuracy(emb_all, labels_all, groups_all)
    print(f"\nLinear Probe Balanced Accuracy: {lp_acc:.4f}")

    # =====================================================================
    # 4. Class-wise Cosine Similarity Matrix
    # =====================================================================
    sim_matrix = _class_cosine_matrix(emb_all, labels_all, num_classes=num_classes)
    print(f"\n--- Cosine Similarity Matrix ({id_label}) ---")
    print("  Diagonal = intra-class mean pairwise, Off-diagonal = inter-class centroid")
    print(np.array2string(sim_matrix, precision=3, suppress_small=True))

    # =====================================================================
    # 5. UMAP Projection
    # =====================================================================
    reducer = umap.UMAP(n_neighbors=50, min_dist=0.01, metric='cosine', random_state=42)
    emb_2d = reducer.fit_transform(emb_all)

    os.makedirs('evidence', exist_ok=True)

    # --- UMAP scatter ---
    fig_umap, ax_umap = plt.subplots(figsize=(10, 8))
    palette = sns.color_palette("husl", num_classes)
    sns.scatterplot(
        x=emb_2d[:, 0], y=emb_2d[:, 1],
        hue=labels_all, palette=palette,
        s=15, alpha=0.7, ax=ax_umap
    )
    ax_umap.set_title(f"UMAP — {id_label}  |  Sil: {sil_score:.3f}  kNN-5: {knn_results[5]:.3f}  LP: {lp_acc:.3f}")
    ax_umap.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
    fig_umap.tight_layout()

    umap_path = f"evidence/umap_loso{loso}.png" if loso else f"evidence/umap_subj{subj}.png"
    fig_umap.savefig(umap_path, dpi=150)
    plt.close(fig_umap)
    print(f"UMAP saved to {umap_path}")

    # --- Cosine similarity heatmap ---
    fig_sim, ax_sim = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        sim_matrix, annot=True, fmt=".3f",
        cmap="RdYlGn", vmin=-0.2, vmax=1.0,
        xticklabels=range(num_classes),
        yticklabels=range(num_classes),
        ax=ax_sim
    )
    ax_sim.set_title(f"Cosine Similarity — {id_label}")
    ax_sim.set_xlabel("Class")
    ax_sim.set_ylabel("Class")
    fig_sim.tight_layout()

    sim_path = f"evidence/cossim_loso{loso}.png" if loso else f"evidence/cossim_subj{subj}.png"
    fig_sim.savefig(sim_path, dpi=150)
    plt.close(fig_sim)
    print(f"Cosine similarity heatmap saved to {sim_path}")

    # =====================================================================
    # W&B Logging
    # =====================================================================
    log_dict = {
        f"Eval/UMAP_{id_label}": wandb.Image(umap_path, caption=f"UMAP {id_label}"),
        f"Eval/CosineSim_{id_label}": wandb.Image(sim_path, caption=f"Cosine Sim {id_label}"),
        f"Eval/Silhouette": sil_score,
        f"Eval/LinearProbe_BalAcc": lp_acc,
        f"Eval/Num_Samples": len(labels_all),
    }
    for k, acc in knn_results.items():
        log_dict[f"Eval/kNN_k{k}_BalAcc"] = acc

    wandb.log(log_dict)
    print(f"Logged to W&B under Eval/ prefix.")