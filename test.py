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
import matplotlib.lines as mlines
import seaborn as sns
import wandb

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# =====================================================================
# Publication-Quality Plot Configuration
# =====================================================================
EMOTION_NAMES = ['Disgust', 'Fear', 'Sad', 'Neutral', 'Happy']

# Curated palette — perceptually distinct, colorblind-friendly, print-safe
_PALETTE_EMO = [
    '#8B5CF6',  # Disgust  — violet
    '#EF4444',  # Fear     — red
    '#3B82F6',  # Sad      — blue
    '#6B7280',  # Neutral  — slate grey
    '#F59E0B',  # Happy    — amber
]

def _apply_pub_style():
    """Set matplotlib rcParams for publication-quality output.

    Font sizes are calibrated so that labels remain legible when the
    figure is shrunk to a single IEEE two-column width (~3.5 in).
    """
    plt.rcParams.update({
        # --- fonts (sized for 3.5" IEEE column) ---
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 13,
        'axes.titlesize': 15,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'legend.title_fontsize': 12,
        # --- figure ---
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        # --- axes ---
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#333333',
        'axes.facecolor': '#FAFAFA',
        'figure.facecolor': 'white',
        # --- ticks ---
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    })

_apply_pub_style()


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

    # 2. Training Set (The other 15 subjects) to act as probe support
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
        # Map string bag_ids directly to integers to match train.py for deterministic CV
        groups_all = np.array([int(b) for b in groups_list])
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

        # 5. Cosine Similarity Heatmap
        suffix = f"loso{loso}" if loso else f"subj{subj}"

        tick_labels = EMOTION_NAMES[:num_classes]
        fig_sim, ax_sim = plt.subplots(figsize=(5.5, 4.8))
        sns.heatmap(
            sim, annot=True, fmt=".3f", annot_kws={'size': 10, 'weight': 'bold'},
            cmap='RdYlGn', vmin=-0.2, vmax=1.0,
            xticklabels=tick_labels, yticklabels=tick_labels,
            square=True, linewidths=1.2, linecolor='white',
            cbar_kws={'shrink': 0.82, 'label': 'Cosine Similarity'},
            ax=ax_sim
        )
        ax_sim.set_title(f"Class Cosine Similarity ({split_name}) — {id_label}",
                         pad=10, fontweight='bold')
        ax_sim.set_xlabel("Emotion Class")
        ax_sim.set_ylabel("Emotion Class")
        ax_sim.tick_params(axis='x', rotation=35)
        ax_sim.tick_params(axis='y', rotation=0)
        fig_sim.tight_layout()

        sim_path = f"evidence/cossim_{split_name.lower()}_{suffix}.png"
        fig_sim.savefig(sim_path)
        plt.close(fig_sim)
        print(f"Cosine similarity heatmap saved to {sim_path}")

        return {
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
        if not _features_are_degenerate(emb_train) and not _features_are_degenerate(emb_val):
            try:
                clf = get_svm()
                clf.fit(emb_train, subj_train)
                subj_cross_acc = balanced_accuracy_score(subj_val, clf.predict(emb_val))
            except Exception as e:
                print(f"  [Proxy-A] Cross-split SVM failed: {e}")
        else:
            print("  [Proxy-A] Skipped cross-split — degenerate features")
        print(f"  Subject ID Prediction (Train->Val cross-split):   {subj_cross_acc:.4f}")
        
        log_dict["Eval/ProxyA_Subject_TrainCV"] = subj_cv_acc
        log_dict["Eval/ProxyA_Subject_TrainVal"] = subj_cross_acc
    else:
        print("  Subject IDs not parsed. Skipping Proxy-A.")

    # =================================================================
    # UMAP Visualizations (3 purpose-built plots)
    # =================================================================
    suffix = f"loso{loso}" if loso else f"subj{subj}"
    print(f"\n{'='*60}")
    print(f"  UMAP 1: Subject Font Proof")
    print(f"{'='*60}")
    # Fit on x_train, transform x_train, color by Subject ID
    reducer1 = make_umap(n_neighbors=50, min_dist=0.01, metric='cosine', random_state=42, n_jobs=-1)
    emb_train_2d_s = reducer1.fit_transform(emb_train)

    unique_train_subjs = np.unique(subj_train)
    n_subjs = len(unique_train_subjs)
    palette_subj = sns.color_palette('husl', n_subjs)

    fig1, ax1 = plt.subplots(figsize=(7.5, 6))
    for i, sid in enumerate(unique_train_subjs):
        mask = subj_train == sid
        ax1.scatter(
            emb_train_2d_s[mask, 0], emb_train_2d_s[mask, 1],
            c=[palette_subj[i]], s=12, alpha=0.65, edgecolors='none',
            label=f'S{sid}', rasterized=True
        )
    ax1.set_title(f"Subject Fingerprint — {id_label}", fontweight='bold', pad=10)
    ax1.set_xlabel('UMAP-1')
    ax1.set_ylabel('UMAP-2')
    ax1.legend(title='Subject', loc='lower right',
               ncol=max(1, n_subjs // 8), frameon=True, framealpha=0.85,
               facecolor='white', edgecolor='#CCCCCC',
               markerscale=1.8, handletextpad=0.3)
    ax1.set_facecolor('#F5F5F5')
    ax1.grid(True, alpha=0.15, linewidth=0.4)
    fig1.tight_layout()
    umap1_path = f"evidence/umap1_subject_font_{suffix}.png"
    fig1.savefig(umap1_path)
    plt.close(fig1)
    log_dict["Eval/UMAP1_SubjectFont"] = wandb.Image(umap1_path, caption="UMAP 1 · Subject Fingerprint")
    print(f"  Saved {umap1_path}")

    print(f"\n{'='*60}")
    print(f"  UMAP 2: Domain Shift (train ○ + val ★)")
    print(f"{'='*60}")
    # Fit on x_train, transform x_train (light circles) and x_val (dark stars), color by Emotion
    reducer2 = make_umap(n_neighbors=50, min_dist=0.01, metric='cosine', random_state=42, n_jobs=-1)
    emb_train_2d_d = reducer2.fit_transform(emb_train)
    emb_val_2d_d = reducer2.transform(emb_val)

    fig2, ax2 = plt.subplots(figsize=(7.5, 6))
    # Train points: semi-transparent circles
    for c in range(num_classes):
        mask_c = labels_train == c
        ax2.scatter(
            emb_train_2d_d[mask_c, 0], emb_train_2d_d[mask_c, 1],
            c=[_PALETTE_EMO[c]], s=10, alpha=0.25, marker='o',
            edgecolors='none', rasterized=True
        )
    # Val points: bold stars with dark edge
    for c in range(num_classes):
        mask_c = labels_val == c
        ax2.scatter(
            emb_val_2d_d[mask_c, 0], emb_val_2d_d[mask_c, 1],
            c=[_PALETTE_EMO[c]], s=70, alpha=0.92, marker='*',
            edgecolors='#222222', linewidths=0.4, rasterized=True
        )
    # Build a proper two-column legend
    legend_handles = []
    for c in range(num_classes):
        legend_handles.append(mlines.Line2D(
            [], [], color=_PALETTE_EMO[c], marker='o', linestyle='None',
            markersize=5, alpha=0.5, label=f'Train — {EMOTION_NAMES[c]}'))
    for c in range(num_classes):
        legend_handles.append(mlines.Line2D(
            [], [], color=_PALETTE_EMO[c], marker='*', linestyle='None',
            markersize=9, markeredgecolor='#222222', markeredgewidth=0.4,
            label=f'Val — {EMOTION_NAMES[c]}'))
    ax2.legend(handles=legend_handles, loc='lower left',
               ncol=2, frameon=True, framealpha=0.85,
               facecolor='white', edgecolor='#CCCCCC',
               handletextpad=0.3, columnspacing=0.8)
    ax2.set_title(f"Domain Shift — {id_label}", fontweight='bold', pad=10)
    ax2.set_xlabel('UMAP-1')
    ax2.set_ylabel('UMAP-2')
    ax2.set_facecolor('#F5F5F5')
    ax2.grid(True, alpha=0.15, linewidth=0.4)
    fig2.tight_layout()
    umap2_path = f"evidence/umap2_domain_shift_{suffix}.png"
    fig2.savefig(umap2_path)
    plt.close(fig2)
    log_dict["Eval/UMAP2_DomainShift"] = wandb.Image(umap2_path, caption="UMAP 2 · Domain Shift")
    print(f"  Saved {umap2_path}")

    print(f"\n{'='*60}")
    print(f"  UMAP 3: Validation Separability")
    print(f"{'='*60}")
    # Fit on x_val, transform x_val, color by Emotion
    reducer3 = make_umap(n_neighbors=50, min_dist=0.01, metric='cosine', random_state=42, n_jobs=-1)
    emb_val_2d_v = reducer3.fit_transform(emb_val)

    fig3, ax3 = plt.subplots(figsize=(7.5, 6))
    for c in range(num_classes):
        mask_c = labels_val == c
        ax3.scatter(
            emb_val_2d_v[mask_c, 0], emb_val_2d_v[mask_c, 1],
            c=[_PALETTE_EMO[c]], s=16, alpha=0.7, edgecolors='none',
            label=EMOTION_NAMES[c], rasterized=True
        )
    ax3.set_title(f"Validation Separability — {id_label}", fontweight='bold', pad=10)
    ax3.set_xlabel('UMAP-1')
    ax3.set_ylabel('UMAP-2')
    ax3.legend(title='Emotion', loc='lower right',
               frameon=True, framealpha=0.85, facecolor='white',
               edgecolor='#CCCCCC', markerscale=1.8)
    ax3.set_facecolor('#F5F5F5')
    ax3.grid(True, alpha=0.15, linewidth=0.4)
    fig3.tight_layout()
    umap3_path = f"evidence/umap3_val_sep_{suffix}.png"
    fig3.savefig(umap3_path)
    plt.close(fig3)
    log_dict["Eval/UMAP3_ValSeparability"] = wandb.Image(umap3_path, caption="UMAP 3 · Validation Separability")
    print(f"  Saved {umap3_path}")

    # =================================================================
    # Intra-Bag Cosine Similarity Heatmaps (sampled from train set)
    # =================================================================
    print(f"\n{'='*60}")
    print(f"  Intra-Bag Cosine Similarity Heatmaps")
    print(f"{'='*60}")

    # Gather all unique bag_ids, sample one, then collect ALL windows for that bag_id
    # (each sample entry is (bag_id, files, emotion_idx, subject_idx, ...))
    bag_id_to_info = {}
    for s in train_set.samples:
        bid = s[0]
        if bid not in bag_id_to_info:
            bag_id_to_info[bid] = {"files": [], "emotion": s[2], "subject": s[3]}
        bag_id_to_info[bid]["files"].extend(s[1])  # accumulate all window files

    def _sample_and_render_intrabag(bag_id_to_info, emotion_filter, exclude_neutral, rng_seed, tag_name, log_dict):
        """Sample a bag matching the filter criteria, extract window embeddings, render heatmap.
        
        Args:
            emotion_filter: If not None, only bags with this emotion are considered.
            exclude_neutral: If True, exclude emotion==3 from candidates.
            rng_seed: Random seed for reproducible sampling.
            tag_name: W&B log key suffix and filename suffix.
        """
        if emotion_filter is not None:
            candidates = [bid for bid, info in bag_id_to_info.items()
                          if len(info["files"]) > 100 and info["emotion"] == emotion_filter]
            if not candidates:
                print(f"  [IntraBag-{tag_name}] No bag with emotion={emotion_filter} and W>100. Falling back to any bag with emotion={emotion_filter}.")
                candidates = [bid for bid, info in bag_id_to_info.items()
                              if info["emotion"] == emotion_filter]
        else:
            candidates = [bid for bid, info in bag_id_to_info.items()
                          if len(info["files"]) > 100 and (not exclude_neutral or info["emotion"] != 3)]
            if not candidates:
                print(f"  [IntraBag-{tag_name}] No matching bag with W>100. Falling back to all bags.")
                candidates = list(bag_id_to_info.keys())

        if not candidates:
            print(f"  [IntraBag-{tag_name}] No bags found at all. Skipping.")
            return

        rng = np.random.RandomState(rng_seed)
        sampled_bag_id = candidates[rng.randint(0, len(candidates))]
        sampled_info = bag_id_to_info[sampled_bag_id]
        sampled_files = sampled_info["files"]
        sampled_emotion = sampled_info["emotion"]
        sampled_subject = sampled_info["subject"]
        print(f"  [{tag_name}] Sampled bag_id={sampled_bag_id} (subject={sampled_subject}, emotion={sampled_emotion}, "
              f"windows={len(sampled_files)})")

        # Load all windows in the bag and extract per-window embeddings
        window_embs = []
        model.eval()
        with torch.no_grad():
            for fname in sampled_files:
                fpath = os.path.join(train_set.samples_dir, fname)
                try:
                    win = torch.load(fpath, weights_only=True, map_location='cpu').float()
                except Exception as e:
                    print(f"    Skipping {fname}: {e}")
                    continue
                # win shape: [T, C, H, W] — add batch dim
                win = win.unsqueeze(0).to(device)            # [1, T, C, H, W]
                win = win.permute(1, 0, 2, 3, 4)             # [T, 1, C, H, W]
                feat, _ = model.encoder(win)                  # encoder output
                if feat.dim() == 5:
                    emb_w = feat.mean(dim=[0, 3, 4])          # [1, D]
                else:
                    emb_w = feat.mean(dim=[-2, -1])           # [1, D]
                emb_w = F.normalize(emb_w, dim=1, eps=1e-6)
                window_embs.append(emb_w.cpu())

        if len(window_embs) >= 2:
            bag_embs = torch.cat(window_embs, dim=0).numpy()  # [W, D]
            bag_cos_sim = cosine_similarity(bag_embs)          # [W, W]

            emo_str = EMOTION_NAMES[sampled_emotion] if sampled_emotion < len(EMOTION_NAMES) else str(sampled_emotion)
            fig_bag, ax_bag = plt.subplots(figsize=(6.5, 5.8))
            sns.heatmap(
                bag_cos_sim, cmap='magma', vmin=-0.1, vmax=1.0,
                square=True, ax=ax_bag,
                xticklabels=False, yticklabels=False,
                linewidths=0, rasterized=True,
                cbar_kws={'shrink': 0.82, 'label': 'Cosine Similarity'}
            )
            ax_bag.set_title(
                f"Intra-Bag Cosine Similarity\n"
                f"Subject {sampled_subject} · {emo_str} · W={len(bag_embs)}",
                fontweight='bold', pad=10
            )
            ax_bag.set_xlabel('Window Index')
            ax_bag.set_ylabel('Window Index')
            fig_bag.tight_layout()
            bag_hm_path = f"evidence/intrabag_cossim_{tag_name}_{suffix}.png"
            fig_bag.savefig(bag_hm_path)
            plt.close(fig_bag)
            log_dict[f"Eval/IntraBag_CosSim_{tag_name}"] = wandb.Image(bag_hm_path, caption=f"Intra-Bag CosSim · {emo_str} (bag {sampled_bag_id})")
            print(f"  Saved {bag_hm_path}")
        else:
            print(f"  [IntraBag-{tag_name}] Not enough windows to compute pairwise similarity.")

    # 1. Non-Neutral emotion bag (original behavior)
    _sample_and_render_intrabag(bag_id_to_info, emotion_filter=None, exclude_neutral=True,
                                rng_seed=42, tag_name="Active", log_dict=log_dict)

    # 2. Neutral emotion bag (emotion_id == 3)
    _sample_and_render_intrabag(bag_id_to_info, emotion_filter=3, exclude_neutral=False,
                                rng_seed=42, tag_name="Neutral", log_dict=log_dict)

    wandb.log(log_dict)
    print(f"\nLogged to W&B under Eval/ and EvalTrain/ prefixes.")