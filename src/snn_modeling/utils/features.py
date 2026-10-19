"""Shared window/bag feature utilities — one implementation, every call site.

Fixes audit B1 (precompute fed windows on the encoder's TIME axis) and the
triplicated extractors (report §2.1). Conventions (fixed.pdf §2):

  - Window tensors on disk: (T, C, H, Wg) = (32 frames, 5 bands, 32, 32).
  - The encoder is TIME-FIRST: input (T, B, C, H, Wg), returns (features, skips)
    with features (T, B, D, 1, 1).
  - Per-window embedding z = GAP over (T, H, W) of the final feature map,
    L2-normalized to the unit sphere S^{d-1} (d = 256).
  - Bag artifact payload schema v2 (see make_bag_payload): consumers must
    validate it; pre-v1/v1 files are refused, never substituted.

Never inline another permute/mean/GAP — import these helpers.
"""

import torch
import torch.nn.functional as F

FEATURE_DIM = 256
BAG_PAYLOAD_SCHEMA = "v2"


def load_encoder_state_strict(encoder, checkpoint, encoder_prefix="encoder."):
    """Load the checkpoint's `encoder.*` parameters into `encoder`, STRICTLY.

    strict=False architecture drift (audit §3.5 rows 24-25) raises here instead:
    missing, unexpected, or shape-mismatched parameters abort the run.
    """
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a checkpoint dict, got {type(checkpoint)}")
    state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    enc_state = {k[len(encoder_prefix):]: v
                 for k, v in state.items() if k.startswith(encoder_prefix)}
    if not enc_state:
        raise RuntimeError(
            f"Checkpoint contains no '{encoder_prefix}*' parameters. "
            f"Top-level key groups: {sorted({k.split('.')[0] for k in state})[:8]}")
    encoder.load_state_dict(enc_state, strict=True)
    return encoder


def encode_windows(encoder, windows, device):
    """(W, T, C, H, Wg) stacked windows -> (W, FEATURE_DIM) L2-normalized z.

    dim0 of `windows` is the window/bag axis, dim1 is each window's time axis.
    The encoder consumes (T, B, C, H, Wg), so the permute below is mandatory —
    omitting it makes windows the time axis and the GAP averages the wrong
    dimension (that was audit bug B1).
    """
    if windows.dim() != 5:
        raise ValueError(
            f"encode_windows expects (W,T,C,H,Wg) 5D stacked windows, got shape {tuple(windows.shape)}")
    n_windows = windows.shape[0]
    features, _ = encoder(windows.permute(1, 0, 2, 3, 4).to(device))
    if features.dim() == 5:
        z = features.mean(dim=[0, 3, 4])          # over T, H, W -> (B=windows, D)
    else:
        raise ValueError(
            f"Encoder returned unexpected {features.dim()}D features of shape {tuple(features.shape)}")
    if z.shape[0] != n_windows:
        raise RuntimeError(
            f"Window-count mismatch: encoded {n_windows} windows into {z.shape[0]} rows — "
            "layout drift somewhere upstream (B1 regression guard)")
    if z.shape[1] != FEATURE_DIM:
        raise RuntimeError(f"Feature dim {z.shape[1]} != FEATURE_DIM {FEATURE_DIM}; "
                           "update features.FEATURE_DIM only with the spec, not silently")
    return F.normalize(z.float(), dim=1, eps=1e-6)


@torch.no_grad()
def encode_bag_batch(encoder, video, device):
    """Encode a dataloader batch of bags into per-bag GAP'd embeddings.

    video: (B, K, T, C, H, Wg) 6D bagged batch or (B, T, C, H, Wg) 5D
    (bagging off, K=1). Returns (B, FEATURE_DIM).
    """
    if video.dim() == 5:
        video = video.unsqueeze(1)               # (B,1,T,C,H,Wg)
    if video.dim() != 6:
        raise ValueError(f"Expected 5D/6D video batch, got {video.dim()}D shape {tuple(video.shape)}")
    B, K = video.shape[0], video.shape[1]
    rows = [encode_windows(encoder, video[b], device) for b in range(B)]   # each (K, D)
    return torch.stack(rows, dim=0).mean(dim=1)


def bag_subject_map(dataset):
    """{str(bag_id): raw_subject_idx} from a SWEEPDataset's samples tuple."""
    return {str(sample[0]): sample[3] for sample in dataset.samples}


def extract_embeddings(loader, encoder, device, desc="Extracting"):
    """Bag-level embeddings over a loader. Returns (N, D) cpu tensor, labels, bag_ids.

    Accepts the 6-tuple (no-aug) or 8-tuple (aug dual-view) raw dataset batches.
    """
    from tqdm import tqdm

    encoder.eval()
    embs, labels, bag_ids = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            video = batch[0]
            lbl = batch[2]
            bag_id = batch[3]
            z_bag = encode_bag_batch(encoder, video, device)
            embs.append(z_bag.cpu())
            labels.append(lbl.cpu() if isinstance(lbl, torch.Tensor) else torch.as_tensor(lbl))
            bag_ids.extend(int(b) for b in bag_id)
    return torch.cat(embs, dim=0), torch.cat(labels, dim=0), bag_ids


def make_bag_payload(z, bag_id, subject, label, window_ids=None):
    """Validate + assemble the v2 per-bag artifact. z: (T_b, FEATURE_DIM) cpu.

    Feature files store ONLY z (L2-normalized backbone embedding) and metadata;
    h', z-tilde, alpha, R_b are recomputed at load from the frozen (mu, U)
    state — artifacts never bake in state that can drift (plan Stage 0.2).
    """
    if z.ndim != 2 or z.shape[1] != FEATURE_DIM:
        raise ValueError(f"bag payload z must be (T_b, {FEATURE_DIM}), got {tuple(z.shape)}")
    if z.shape[0] == 0:
        raise ValueError(f"bag {bag_id}: empty window count")
    norms = z.norm(dim=1)
    if not torch.isfinite(z).all() or (norms - 1.0).abs().max() > 1e-3:
        raise ValueError(f"bag {bag_id}: payload z must be finite and L2-normalized per row "
                         f"(norm range [{norms.min():.4f}, {norms.max():.4f}])")
    return {
        'schema': BAG_PAYLOAD_SCHEMA,
        'z': z.float().cpu(),
        'bag_id': int(bag_id),
        'subject': int(subject),
        'label': int(label),
        'T_b': int(z.shape[0]),
        'window_ids': torch.as_tensor(
            window_ids if window_ids is not None else list(range(int(z.shape[0]))),
            dtype=torch.long),
    }


def load_bag_payload(path):
    """Load + validate a v2 bag artifact. Raises on legacy/corrupt files — no substitution."""
    obj = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(obj, dict):
        raise RuntimeError(
            f"{path} is a legacy pre-v2 bag artifact (bare tensor; the B1 time-axis-swap bug "
            "era). Delete precomputed_features/ and regenerate with the fixed precompute_features.py.")
    if obj.get('schema') != BAG_PAYLOAD_SCHEMA:
        raise RuntimeError(
            f"{path}: schema={obj.get('schema')!r} != {BAG_PAYLOAD_SCHEMA!r}; regenerate.")
    z = obj['z']
    if z.shape[0] != obj['T_b'] or z.shape[1] != FEATURE_DIM:
        raise RuntimeError(f"{path}: payload inconsistent (rows={tuple(z.shape)}, T_b={obj['T_b']})")
    return obj
