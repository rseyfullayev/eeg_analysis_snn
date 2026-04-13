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


def _knn_accuracy(emb_train, labels_train, emb_val, labels_val, k_values=(1, 3, 5, 10)):
    """k-NN accuracy evaluating on Val set using Train set as support."""
    results = {}
    
    for k in k_values:
        clf = KNeighborsClassifier(n_neighbors=k, metric='cosine')
        clf.fit(emb_train, labels_train)
        preds_val = clf.predict(emb_val)
        
        bal_acc = balanced_accuracy_score(labels_val, preds_val)
        results[k] = bal_acc

    return results


def _linear_probe_accuracy(emb_train, labels_train, emb_val, labels_val):
    """Logistic regression linear probe trained on one split, evaluated on another."""
    clf = LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        multi_class='multinomial',
        C=1.0
    )
    clf.fit(emb_train, labels_train)
    preds_val = clf.predict(emb_val)

    return balanced_accuracy_score(labels_val, preds_val)


def _linear_probe_cv(emb, labels, groups, n_folds=5):
    """Intra-split linear probe with StratifiedGroupKFold to prevent session leakage."""
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

    # ------------------
    # Data Loaders
    # ------------------
    masks = torch.load(os.path.join(config.data.dataset_path, 'masks.pt'), weights_only=True).to(device)

    # 1. Validation Set (The held-out LOSO subject)
    val_set = SWEEPDataset(
        config,
        split='val',
        loso=loso,
        subj=subj,
        prototypes=masks
    )

    # 2. Training Set (The other 14 subjects) to act as probe support
    train_set = SWEEPDataset(
        config,
        split='train',
        loso=loso,
        subj=subj,
        prototypes=masks
    )

    num_workers = config.data.get('num_workers', 0)
    prefetch = config.data.get('prefetch_factor', 2) if num_workers > 0 else None
    persist = num_workers > 0

    val_loader = DataLoader(
        val_set, batch_size=config.training.batch_size, shuffle=False,
        num_workers=num_workers, prefetch_factor=prefetch, persistent_workers=persist, pin_memory=True
    )
    
    train_loader = DataLoader(
        train_set, batch_size=config.training.batch_size, shuffle=False,
        num_workers=num_workers, prefetch_factor=prefetch, persistent_workers=persist, pin_memory=True
    )

    # ------------------
    # Extraction Helper
    # ------------------
    from tqdm import tqdm
    def extract_embs(loader, desc="Extracting"):
        embs, labels_list, groups_list = [], [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc):
                vid = batch[0]
                lbl = batch[2] if len(batch) >= 4 else batch[1]
                bag_id = batch[-1]  # bag_id is always the last element
                
                K_bag = None
                if vid.dim() == 6:
                    B, K_bag, T, C, H, W = vid.shape
                    vid = vid.view(B * K_bag, T, C, H, W)
                else:
                    B, T, C, H, W = vid.shape

                vid = vid.permute(1, 0, 2, 3, 4).to(device)
                features, _ = model.encoder(vid)

                if features.dim() == 5:
                    emb = features.mean(dim=[0, 3, 4])
                else:
                    emb = features.mean(dim=[-2, -1])

                if K_bag is not None and emb.shape[0] == B * K_bag:
                    emb = emb.view(B, K_bag, -1).mean(dim=1)

                emb = F.normalize(emb, dim=1, eps=1e-6)

                embs.append(emb.cpu())
                labels_list.append(lbl)
                groups_list.extend(bag_id)

        emb_all = torch.cat(embs, dim=0).numpy()
        labels_all = torch.cat(labels_list, dim=0).numpy()
        # Map string bag_ids to integer groups for CV
        unique_groups = {b: i for i, b in enumerate(set(groups_list))}
        groups_all = np.array([unique_groups[b] for b in groups_list])
        return emb_all, labels_all, groups_all

    print("Extracting Validation Embeddings...")
    emb_val, labels_val, groups_val = extract_embs(val_loader, desc="Val Embeddings")
    
    print("Extracting Train (Support) Embeddings...")
    emb_train, labels_train, groups_train = extract_embs(train_loader, desc="Train Embeddings")

    num_classes = config.model.get('num_classes', 5)
    id_label = f"LOSO {loso}" if loso else f"Subj {subj}"

    print(f"Val Embeddings: {emb_val.shape} | Train Embeddings: {emb_train.shape}")

    os.makedirs('evidence', exist_ok=True)

    # =================================================================
    # Reusable evaluation for a single split
    # =================================================================
    def _evaluate_split(emb, labels, emb_support, labels_support, split_name, tag):
        """Run all metrics for one split.

        Args:
            emb / labels:                   embeddings & labels to evaluate ON.
            emb_support / labels_support:   embeddings & labels to FIT probes ON
                                            (for kNN / Linear Probe).
            split_name:  human-readable name, e.g. "Val" or "Train".
            tag:         W&B prefix, e.g. "Eval" or "EvalTrain".
        """
        print(f"\n{'='*60}")
        print(f"  {split_name} Set Evaluation  ({id_label})")
        print(f"{'='*60}")

        # 1. Silhouette
        sil = silhouette_score(emb, labels, metric='cosine') if len(set(labels)) > 1 else 0.0
        print(f"Silhouette Score (cosine): {sil:.4f}")

        # 2. k-NN
        knn = _knn_accuracy(emb_support, labels_support, emb, labels, k_values=(1, 3, 5, 10))
        print(f"\n--- k-NN Balanced Accuracy ---")
        for k, acc in knn.items():
            print(f"  k={k:>2d}: {acc:.4f}")

        # 3. Linear Probe
        lp = _linear_probe_accuracy(emb_support, labels_support, emb, labels)
        print(f"\nLinear Probe Balanced Accuracy: {lp:.4f}")

        # 4. Cosine Similarity Matrix
        sim = _class_cosine_matrix(emb, labels, num_classes=num_classes)
        print(f"\n--- Cosine Similarity Matrix ---")
        print("  Diagonal = intra-class mean pairwise, Off-diagonal = inter-class centroid")
        print(np.array2string(sim, precision=3, suppress_small=True))

        # 5. UMAP
        reducer = umap.UMAP(n_neighbors=50, min_dist=0.01, metric='cosine', random_state=42)
        emb_2d = reducer.fit_transform(emb)

        palette = sns.color_palette("husl", num_classes)

        fig_umap, ax_umap = plt.subplots(figsize=(10, 8))
        sns.scatterplot(
            x=emb_2d[:, 0], y=emb_2d[:, 1],
            hue=labels, palette=palette,
            s=15, alpha=0.7, ax=ax_umap
        )
        ax_umap.set_title(f"UMAP {split_name} — {id_label}  |  Sil: {sil:.3f}  kNN-5: {knn[5]:.3f}  LP: {lp:.3f}")
        ax_umap.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
        fig_umap.tight_layout()

        suffix = f"loso{loso}" if loso else f"subj{subj}"
        umap_path = f"evidence/umap_{split_name.lower()}_{suffix}.png"
        fig_umap.savefig(umap_path, dpi=150)
        plt.close(fig_umap)
        print(f"UMAP saved to {umap_path}")

        fig_sim, ax_sim = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            sim, annot=True, fmt=".3f",
            cmap="RdYlGn", vmin=-0.2, vmax=1.0,
            xticklabels=range(num_classes),
            yticklabels=range(num_classes),
            ax=ax_sim
        )
        ax_sim.set_title(f"Cosine Similarity {split_name} — {id_label}")
        ax_sim.set_xlabel("Class")
        ax_sim.set_ylabel("Class")
        fig_sim.tight_layout()

        sim_path = f"evidence/cossim_{split_name.lower()}_{suffix}.png"
        fig_sim.savefig(sim_path, dpi=150)
        plt.close(fig_sim)
        print(f"Cosine similarity heatmap saved to {sim_path}")

        return {
            f"{tag}/UMAP_{id_label}": wandb.Image(umap_path, caption=f"UMAP {split_name} {id_label}"),
            f"{tag}/CosineSim_{id_label}": wandb.Image(sim_path, caption=f"Cosine Sim {split_name} {id_label}"),
            f"{tag}/Silhouette": sil,
            f"{tag}/LinearProbe_BalAcc": lp,
            f"{tag}/Num_Samples": len(labels),
            **{f"{tag}/kNN_k{k}_BalAcc": acc for k, acc in knn.items()},
        }

    # =================================================================
    # Run evaluation on BOTH splits
    # =================================================================
    # Val:   fit probes on Train, evaluate on Val
    log_val = _evaluate_split(emb_val, labels_val, emb_train, labels_train, "Val", "Eval")

    # Train: fit probes on Val, evaluate on Train (symmetric sanity check)
    log_train = _evaluate_split(emb_train, labels_train, emb_val, labels_val, "Train", "EvalTrain")

    # =================================================================
    # 4-Way Linear Probe
    # =================================================================
    print(f"\n{'='*60}")
    print(f"  4-Way Linear Probe  ({id_label})")
    print(f"{'='*60}")

    # 1. Train→Train (intra-split CV with bag_id groups)
    lp_train_train = _linear_probe_cv(emb_train, labels_train, groups_train, n_folds=5)
    print(f"  Train→Train (5-fold GroupCV): {lp_train_train:.4f}")

    # 2. Val→Val (intra-split CV with bag_id groups)
    lp_val_val = _linear_probe_cv(emb_val, labels_val, groups_val, n_folds=5)
    print(f"  Val→Val   (5-fold GroupCV):   {lp_val_val:.4f}")

    # 3. Train→Val (cross-split)
    lp_train_val = _linear_probe_accuracy(emb_train, labels_train, emb_val, labels_val)
    print(f"  Train→Val (cross-split):      {lp_train_val:.4f}")

    # 4. Val→Train (cross-split)
    lp_val_train = _linear_probe_accuracy(emb_val, labels_val, emb_train, labels_train)
    print(f"  Val→Train (cross-split):      {lp_val_train:.4f}")

    # =====================================================================
    # W&B Logging
    # =====================================================================
    log_dict = {**log_val, **log_train}
    log_dict["Eval/Num_Train_Support_Samples"] = len(labels_train)
    log_dict["Eval/Num_Val_Samples"] = len(labels_val)
    log_dict["Eval/LP_TrainTrain_CV"] = lp_train_train
    log_dict["Eval/LP_ValVal_CV"] = lp_val_val
    log_dict["Eval/LP_TrainVal"] = lp_train_val
    log_dict["Eval/LP_ValTrain"] = lp_val_train

    wandb.log(log_dict)
    print(f"\nLogged to W&B under Eval/ and EvalTrain/ prefixes.")