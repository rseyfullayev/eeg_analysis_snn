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

try:
    from cuml.manifold import UMAP as cuUMAP
    from cuml.linear_model import LogisticRegression as cuLogisticRegression
    from cuml.svm import SVC as cuSVC
    _USE_CUML = True
    print("[test.py] cuML UMAP/LogReg/SVM detected — using GPU-accelerated variants.")
except ImportError:
    import umap.umap_ as umap
    _USE_CUML = False
    print("[test.py] cuML not available — falling back to CPU scikit-learn.")

def _features_are_degenerate(emb):
    """Check if features are degenerate (NaN, Inf, or zero variance across all columns).
    
    This happens during representational collapse (common early in Muon training).
    cuML's SVM computes gamma = 1 / (n_cols * x_var), which causes ZeroDivisionError
    when variance is zero. cuML LogReg L-BFGS also produces NaN on degenerate inputs.
    """
    if np.any(np.isnan(emb)) or np.any(np.isinf(emb)):
        return True
    # Check if ALL features have zero variance (representational collapse)
    col_var = np.var(emb, axis=0)
    if np.all(col_var < 1e-12):
        return True
    return False

def get_svm(**kwargs):
    if _USE_CUML:
        try:
            from cuml.svm import LinearSVC as cuLinearSVC
            return cuLinearSVC(max_iter=1000)
        except ImportError:
            return cuSVC(kernel='linear', gamma='scale')
    else:
        from sklearn.svm import LinearSVC
        return LinearSVC(max_iter=1000)

def get_logistic_regression(**kwargs):
    if _USE_CUML:
        return cuLogisticRegression(max_iter=1000)
    else:
        return LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial', C=1.0)

def make_umap(**kwargs):
    """Factory that returns a UMAP reducer using cuML (GPU) if available, else CPU umap-learn."""
    if _USE_CUML:
        # cuML UMAP doesn't support n_jobs or 'cosine' via string the same way;
        # it uses metric='cosine' fine, but no n_jobs param.
        kwargs.pop('n_jobs', None)
        return cuUMAP(**kwargs)
    else:
        return umap.UMAP(**kwargs)

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
    if _features_are_degenerate(emb_train) or _features_are_degenerate(emb_val):
        print("  [LinearProbe] Skipped — degenerate features (NaN/Inf/zero-var)")
        return 0.0
    try:
        clf = get_logistic_regression()
        clf.fit(emb_train, labels_train)
        preds_val = clf.predict(emb_val)
        return balanced_accuracy_score(labels_val, preds_val)
    except Exception as e:
        print(f"  [LinearProbe] Failed: {e}")
        return 0.0


def _linear_probe_cv(emb, labels, groups, n_folds=5):
    """Intra-split linear probe with StratifiedGroupKFold to prevent session leakage."""
    if _features_are_degenerate(emb):
        print("  [LinearProbeCV] Skipped — degenerate features (NaN/Inf/zero-var)")
        return 0.0
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    preds_all, labels_all = [], []

    for train_idx, test_idx in sgkf.split(emb, labels, groups=groups):
        try:
            clf = get_logistic_regression()
            clf.fit(emb[train_idx], labels[train_idx])
            preds_all.append(clf.predict(emb[test_idx]))
            labels_all.append(labels[test_idx])
        except Exception as e:
            print(f"  [LinearProbeCV] Fold failed: {e}")
            continue

    if len(preds_all) == 0:
        return 0.0
    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)
    return balanced_accuracy_score(labels_all, preds_all)

def _svm_probe_accuracy(emb_train, labels_train, emb_val, labels_val):
    """Linear SVM probe trained on one split, evaluated on another."""
    if _features_are_degenerate(emb_train) or _features_are_degenerate(emb_val):
        print("  [SVMProbe] Skipped — degenerate features (NaN/Inf/zero-var)")
        return 0.0
    try:
        clf = get_svm()
        clf.fit(emb_train, labels_train)
        preds_val = clf.predict(emb_val)
        return balanced_accuracy_score(labels_val, preds_val)
    except Exception as e:
        print(f"  [SVMProbe] Failed: {e}")
        return 0.0

def _svm_probe_cv(emb, labels, groups, n_folds=5):
    """Intra-split linear SVM probe with StratifiedGroupKFold."""
    if _features_are_degenerate(emb):
        print("  [SVMProbeCV] Skipped — degenerate features (NaN/Inf/zero-var)")
        return 0.0
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    preds_all, labels_all = [], []

    for train_idx, test_idx in sgkf.split(emb, labels, groups=groups):
        try:
            clf = get_svm()
            clf.fit(emb[train_idx], labels[train_idx])
            preds_all.append(clf.predict(emb[test_idx]))
            labels_all.append(labels[test_idx])
        except Exception as e:
            print(f"  [SVMProbeCV] Fold failed: {e}")
            continue

    if len(preds_all) == 0:
        return 0.0
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
        model: A SpikingMobileNetProjector (or anything with an .encoder attribute).
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
    # Pre-build a fast lookup dictionary: bag_id -> subject_idx
    # In dataset.py, samples is a list of (bag_id, files, emotion_idx, subject_idx, ...)
    val_bag_to_subj = {str(item[0]): item[3] for item in val_set.samples}
    train_bag_to_subj = {str(item[0]): item[3] for item in train_set.samples}
    bag_to_subj = {**train_bag_to_subj, **val_bag_to_subj}

    from tqdm import tqdm
    def extract_embs(loader, desc="Extracting"):
        embs, labels_list, groups_list, subjects_list = [], [], [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc):
                vid = batch[0]
                lbl = batch[2] if len(batch) >= 4 else batch[1]
                bag_id = batch[3]  # bag_id is always at index 3
                
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
                
                # Lookup subjects
                subjects = []
                for b in bag_id:
                    # PyTorch dataloaders return tensors for integers, so unwrap them
                    key = str(b.item() if hasattr(b, 'item') else b)
                    subjects.append(bag_to_subj.get(key, -1))

                subjects_list.extend(subjects)

        emb_all = torch.cat(embs, dim=0).numpy()
        labels_all = torch.cat(labels_list, dim=0).numpy()
        # Map string bag_ids to integer groups for CV
        unique_groups = {b: i for i, b in enumerate(set(groups_list))}
        groups_all = np.array([unique_groups[b] for b in groups_list])
        subjects_all = np.array(subjects_list)
        return emb_all, labels_all, groups_all, subjects_all

    print("Extracting Validation Embeddings...")
    emb_val, labels_val, groups_val, subj_val = extract_embs(val_loader, desc="Val Embeddings")
    
    print("Extracting Train (Support) Embeddings...")
    emb_train, labels_train, groups_train, subj_train = extract_embs(train_loader, desc="Train Embeddings")

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
        knn = {} # _knn_accuracy(emb_support, labels_support, emb, labels, k_values=(1, 3, 5, 10))
        # print(f"\n--- k-NN Balanced Accuracy ---")
        # for k, acc in knn.items():
        #     print(f"  k={k:>2d}: {acc:.4f}")

        # 3. Linear Probe
        lp = _linear_probe_accuracy(emb_support, labels_support, emb, labels)
        print(f"\nLinear Probe Balanced Accuracy: {lp:.4f}")

        # 4. Cosine Similarity Matrix
        sim = _class_cosine_matrix(emb, labels, num_classes=num_classes)
        print(f"\n--- Cosine Similarity Matrix ---")
        print("  Diagonal = intra-class mean pairwise, Off-diagonal = inter-class centroid")
        print(np.array2string(sim, precision=3, suppress_small=True))

        # 5. UMAP (cuML GPU if available, else CPU)
        reducer = make_umap(n_neighbors=50, min_dist=0.01, metric='cosine', random_state=42, n_jobs=-1)
        emb_2d = reducer.fit_transform(emb)

        palette = sns.color_palette("husl", num_classes)

        fig_umap, ax_umap = plt.subplots(figsize=(10, 8))
        sns.scatterplot(
            x=emb_2d[:, 0], y=emb_2d[:, 1],
            hue=labels, palette=palette,
            s=15, alpha=0.7, ax=ax_umap
        )
        ax_umap.set_title(f"UMAP {split_name} — {id_label}")
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
    # =================================================================
    # Proxy-A Subject ID Probe
    # =================================================================
    print(f"\n{'='*60}")
    print(f"  Proxy-A Subject ID Probe  ({id_label})")
    print(f"{'='*60}")
    
    if not np.all(subj_train == -1):
        # 1. Evaluate predictability of Subject ID inside the Train set.
        # High accuracy = Subject ID is highly preserved. Low accuracy = Subject ID destroyed.
        subj_cv_acc = _svm_probe_cv(emb_train, subj_train, groups_train, n_folds=5)
        print(f"  Subject SVM Prediction (Train Support, 5-fold CV): {subj_cv_acc:.4f}")
        
        # 2. Fit on all Train subjects -> Predict on Test/Val set 
        # (Sanity check: Accuracy will inevitably be low for an unseen test subject)
        subj_cross_acc = 0.0
        clf = None
        if not _features_are_degenerate(emb_train) and not _features_are_degenerate(emb_val):
            try:
                clf = get_svm()
                clf.fit(emb_train, subj_train)
                subj_cross_acc = balanced_accuracy_score(subj_val, clf.predict(emb_val))
            except Exception as e:
                print(f"  [Proxy-A] Cross-split SVM failed: {e}")
                clf = None
        else:
            print("  [Proxy-A] Skipped cross-split — degenerate features")
        print(f"  Subject ID Prediction (Train->Val cross-split):   {subj_cross_acc:.4f}")
        
        log_dict["Eval/ProxyA_Subject_TrainCV"] = subj_cv_acc
        log_dict["Eval/ProxyA_Subject_TrainVal"] = subj_cross_acc

        # =================================================================
        # Subject-Heterogeneous UMAP Generative Probes
        # =================================================================
        print(f"\n{'='*60}")
        print(f"  Subject-Heterogeneous UMAPs")
        print(f"{'='*60}")

        train_reducer = make_umap(n_neighbors=50, min_dist=0.01, metric='cosine', random_state=42, n_jobs=-1)
        emb_train_2d_subj = train_reducer.fit_transform(emb_train)
        
        unique_train_subjs = np.unique(subj_train)
        palette_subj_train = sns.color_palette("husl", len(unique_train_subjs))
        
        fig_subj_tr, ax_subj_tr = plt.subplots(figsize=(10, 8))
        sns.scatterplot(
            x=emb_train_2d_subj[:, 0], y=emb_train_2d_subj[:, 1],
            hue=subj_train, palette=palette_subj_train,
            s=15, alpha=0.7, ax=ax_subj_tr, legend='full'
        )
        ax_subj_tr.set_title(f"Train UMAP Colored by Subject ID — {id_label}")
        ax_subj_tr.legend(title='Subject ID', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        fig_subj_tr.tight_layout()
        
        suffix = f"loso{loso}" if loso else f"subj{subj}"
        train_subj_umap_path = f"evidence/umap_train_subject_{suffix}.png"
        fig_subj_tr.savefig(train_subj_umap_path, dpi=150)
        plt.close(fig_subj_tr)
        log_dict["Eval/UMAP_Subject_Train"] = wandb.Image(train_subj_umap_path, caption=f"UMAP Train by Subject")
        print(f"  Saved {train_subj_umap_path}")
        
        # We need preds_subj_val from the SVM for the val UMAP plot
        if clf is not None:
            preds_subj_val = clf.predict(emb_val)

            # Transform Val using Train-fitted UMAP, colored by Predicted TRAIN Subject IDs
            emb_val_2d_subj = train_reducer.transform(emb_val)
            
            fig_subj_val, ax_subj_val = plt.subplots(figsize=(10, 8))
            sns.scatterplot(
                x=emb_val_2d_subj[:, 0], y=emb_val_2d_subj[:, 1],
                hue=preds_subj_val, palette=palette_subj_train, hue_order=unique_train_subjs,
                s=15, alpha=0.7, ax=ax_subj_val, legend='full'
            )
            ax_subj_val.set_title(f"Val UMAP (Train-Fitted) Colored by PRED Train Subj ID — {id_label}")
            ax_subj_val.legend(title='Pred Train Subj', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
            fig_subj_val.tight_layout()
            
            val_subj_umap_path = f"evidence/umap_val_subject_{suffix}.png"
            fig_subj_val.savefig(val_subj_umap_path, dpi=150)
            plt.close(fig_subj_val)
            log_dict["Eval/UMAP_Subject_Val"] = wandb.Image(val_subj_umap_path, caption=f"UMAP Val by Subject (Train-Fitted trans.)")
            print(f"  Saved {val_subj_umap_path}")
        else:
            print("  [Proxy-A] Skipping Val Subject UMAP — SVM not fitted")

        
    else:
        print("  Subject IDs not parsed. Skipping Proxy-A.")
    wandb.log(log_dict)
    print(f"\nLogged to W&B under Eval/ and EvalTrain/ prefixes.")