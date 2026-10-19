"""Offline T0 sanity + projection-rank harness on a frozen Phase 1A encoder.

Rewritten in Stage 0.3 of the audit plan (REVIEW_SMC_AUDIT.md §5). Fixes audit B2:
the old script imported get_encoder (which does not exist), raw-loaded a Hydra
config (so config.model/config.data were absent post-compose), ignored the
subject-probe half of the rank acceptance criterion, printed the winner without
persisting it, and never seeded RNGs. All shared logic now lives in
utils/features.py + utils/config_io.py (the old triplicated extractor is gone).

Protocol note: the old manual T5 rank sweep is superseded by the dynamic-m rule
(fixed.pdf Rm 6.4 / §11.4(ii)); this harness VERIFIES that rule — it reports the
85/90/99% energy ranks and evaluates the projection at both halves of the owner's
acceptance test (emotion retention AND subject-leakage drop), persisting results
to JSON. Refuses to silently pick a rank when no candidate meets the criteria.

Usage:
    python run_phase1b_t0_t5.py --config config/ea_fda_vib_1b.yaml \
        --ckpt <phase1A checkpoint> [--out t0_t5_results.json] \
        [--emo-tol 0.02] [--sub-drop 0.15]
"""

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestCentroid
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.snn_modeling.dataloader.dataset import SWEEPDataset
from src.snn_modeling.utils.config_io import load_config
from src.snn_modeling.utils.features import bag_subject_map, encode_bag_batch, load_encoder_state_strict
from src.snn_modeling.utils.model_builder import build_model
from test import get_svm


def extract_bag_embeddings(loader, encoder, device):
    """Returns (X, y, subj): per-sample GAP'd L2-normalized embeddings, labels,
    and RAW subject ids (via the dataset's own samples table — no string hacks)."""
    encoder.eval()
    embs, labels, bag_ids = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting"):
            video, _tmap, lbl, bag_id = batch[0], batch[1], batch[2], batch[3]
            z = encode_bag_batch(encoder, video.to(device), device)
            embs.append(z.cpu().numpy())
            labels.append(lbl.cpu().numpy() if isinstance(lbl, torch.Tensor) else np.asarray(lbl))
            bag_ids.extend(int(b) for b in bag_id)
    X = np.concatenate(embs, axis=0).astype(np.float32)
    y = np.concatenate(labels, axis=0)
    return X, y, np.array(bag_ids, dtype=np.int64)


def subspace_project(X, eigvecs, r):
    if r <= 0:
        return X
    P = np.eye(X.shape[1]) - eigvecs[:, :r] @ eigvecs[:, :r].T
    return X @ P


def balanced_cv_accuracy(X, y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(X, y):
        clf = get_svm()
        clf.fit(X[tr], y[tr])
        accs.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(accs))


def run(config_path, ckpt_path, out_path, emo_tol, sub_drop):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config(config_path)  # Hydra compose + architecture->model alias + seed_everything

    print("Loading frozen Phase 1A encoder...")
    model = build_model(config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    # Strict: shape/name drift aborts (the old strict=False silently accepted it).
    encoder = load_encoder_state_strict(model.encoder, ckpt).eval()

    # split='all' loads every subject's bags; the caller partitions by subject id
    # (no more loso='-999' sentinel hack). prototypes=None is fine here: the
    # target_map channel is never used by this harness.
    dataset = SWEEPDataset(config, split='all', prototypes=None, precomputed=False)
    num_workers = config.data.get('num_workers', 0)
    prefetch = config.data.get('prefetch_factor', 2) if num_workers > 0 else None
    loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=False,
                        num_workers=num_workers, prefetch_factor=prefetch, pin_memory=True)
    X, y, bag_ids = extract_bag_embeddings(loader, encoder, device)

    b2s = bag_subject_map(dataset)
    subj = np.array([b2s[str(b)] for b in bag_ids], dtype=np.int64)
    if (subj == -1).any():
        raise RuntimeError("Unmapped subject id found in dataset.samples — B7 regression; aborting.")

    mask_test, mask_val, mask_train = subj == 1, subj == 2, subj > 2
    if not (mask_test.any() and mask_val.any() and mask_train.any()):
        raise RuntimeError(
            f"Protocol requires subjects 1 (outer test), 2 (inner val), >2 (inner train); "
            f"subjects present: {sorted(set(int(s) for s in subj))}")

    Xtr, ytr, subj_tr = X[mask_train], y[mask_train], subj[mask_train]
    K = int(len(np.unique(subj_tr)))

    # ---------------- T0: nearest-centroid sanity on unseen subject 1 ----------------
    nc = NearestCentroid().fit(Xtr, ytr)
    t0 = float(balanced_accuracy_score(y[mask_test], nc.predict(X[mask_test])))
    print(f"T0: nearest-centroid balanced acc on unseen subject 1 = {t0:.4f}")

    # ---------------- Between-subject spectral reference (per-subject means) --------
    mu = Xtr.mean(axis=0)
    S_between = np.zeros((Xtr.shape[1], Xtr.shape[1]), dtype=np.float64)
    for s in np.unique(subj_tr):
        m = subj_tr == s
        v = Xtr[m].mean(axis=0) - mu
        S_between += int(m.sum()) * np.outer(v, v)
    S_between /= len(Xtr)
    vals, vecs = np.linalg.eigh(S_between)
    order = np.argsort(vals)[::-1]
    vals, vecs = np.clip(vals[order], 0.0, None), vecs[:, order]
    total_var = float(vals.sum())
    if total_var <= 0:
        raise RuntimeError("Between-subject scatter is zero — nothing to project; aborting.")
    cum = np.cumsum(vals) / total_var
    rank85 = int(np.argmax(cum >= 0.85) + 1)
    rank90 = int(np.argmax(cum >= 0.90) + 1)
    rank99 = int(np.argmax(cum >= 0.99) + 1)
    print(f"Spectral ranks: 85%={rank85}  90%={rank90}  99%={rank99}  |  max rank S-1 = {K - 1}")
    print(f"Top eigenvalues: {np.round(vals[:min(12, len(vals))], 4).tolist()}")

    # ---------------- Rank sweep with BOTH acceptance criteria ----------------
    candidates = sorted({r for r in [0, rank85, rank90, rank99 - 1, rank99, rank99 + 1, K - 1]
                         if 0 <= r <= K - 1})
    results = {}
    for r in candidates:
        Xtr_p = subspace_project(Xtr, vecs, r)
        Xval_p = subspace_project(X[mask_val], vecs, r)
        clf = get_svm()
        clf.fit(Xtr_p, ytr)
        m_emo = float(balanced_accuracy_score(y[mask_val], clf.predict(Xval_p)))
        a_sub = balanced_cv_accuracy(Xtr_p, subj_tr)
        results[r] = {'M_emo_inner_val': m_emo, 'A_sub_cv': a_sub}
        print(f"  r={r:>3}:  M_emo(inner-val)={m_emo:.4f}   A_sub(subject CV)={a_sub:.4f}")

    m0 = results[0]['M_emo_inner_val']
    s0 = results[0]['A_sub_cv']
    accepted = [r for r in candidates
                if (m0 - results[r]['M_emo_inner_val']) <= emo_tol
                and (s0 - results[r]['A_sub_cv']) >= sub_drop]
    winner = max(accepted) if accepted else None

    payload = {
        'config': config_path,
        'checkpoint': ckpt_path,
        'protocol': {'outer_test_subject': 1, 'inner_val_subject': 2, 'inner_train_subjects': K},
        'T0_nearest_centroid_test_balanced_acc': t0,
        'spectral_ranks': {'rank85': rank85, 'rank90': rank90, 'rank99': rank99,
                           'max_rank_S_minus_1': K - 1},
        'criteria': {'emo_tol': emo_tol, 'sub_drop': sub_drop},
        'baseline': {'M_emo_r0': m0, 'A_sub_r0': s0},
        'per_rank': {str(r): results[r] for r in candidates},
        'winner_r': winner,
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f"Results persisted to {out_path}")

    if winner is None:
        raise RuntimeError(
            f"No candidate rank {candidates} meets BOTH criteria "
            f"(emotion drop <= {emo_tol} AND subject-probe drop >= {sub_drop}). "
            "Refusing to silently pick a rank — inspect the JSON before proceeding.")
    print(f"Winner rank r* = {winner} (largest rank satisfying both criteria)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="T0 sanity + projection-rank verification harness")
    parser.add_argument('--config', type=str, required=True, help='Composed-ready config YAML')
    parser.add_argument('--ckpt', type=str, required=True, help='Phase 1A frozen checkpoint')
    parser.add_argument('--out', type=str, default='t0_t5_results.json')
    parser.add_argument('--emo-tol', type=float, default=0.02, help='max M_emo drop vs r=0')
    parser.add_argument('--sub-drop', type=float, default=0.15, help='min A_sub drop vs r=0')
    args = parser.parse_args()
    run(args.config, args.ckpt, args.out, args.emo_tol, args.sub_drop)
