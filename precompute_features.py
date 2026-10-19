"""Regenerate Phase 1B bag artifacts (schema v2) from a frozen Phase 1A checkpoint.

Fixes audit B1: the encoder is TIME-FIRST, so stacked windows (W, T, C, H, Wg)
must be permuted to (T, W, C, H, Wg) before the forward pass and the GAP must
average over the encoder's time axis — the old code skipped the permute and the
mean collapsed the window axis instead, producing frame-averaged garbage.

Artifacts store ONLY per-window z (L2-normalized 256-D backbone embedding,
fixed.pdf §2) + metadata (subject, label, T_b, window_ids). h', z-tilde,
alpha^(1B) and R_b are recomputed at load from the frozen (mu, U) state, so
feature files never bake in state that can change (plan Stage 0.2).
"""

import argparse
import os
import torch
from tqdm import tqdm

from src.snn_modeling.dataloader.dataset import SWEEPDataset
from src.snn_modeling.utils.config_io import load_config
from src.snn_modeling.utils.features import (encode_windows, load_encoder_state_strict,
                                             make_bag_payload)
from src.snn_modeling.utils.model_builder import build_model


def run_precompute(config, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Building model...")
    model = build_model(config).to(device)
    # Strict load: shape/name mismatches abort (the old strict=False hid drift;
    # audit §3.5 row 25).
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    encoder = load_encoder_state_strict(model.encoder, checkpoint)
    encoder.eval()

    # split='all': every subject's bags (this is feature generation, not evaluation).
    dataset = SWEEPDataset(config, split='all', precomputed=False)
    if not dataset.bagging:
        raise RuntimeError(
            f"precompute requires training.bagging=True (per-bag artifacts); "
            f"config {getattr(config, 'experiment_name', '?')} has bagging=False")

    out_dir = os.path.join(dataset.samples_dir, "precomputed_features")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Precomputing {len(dataset)} bags...")
    window_counts = set()
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Precomputing"):
            bag_id, files, label_idx, subject_idx, _video_id, _unused = dataset.samples[idx]

            # ALL windows of the bag — no strided subsampling at encoding time;
            # missing/corrupt files abort the run with the path (audit §3.5 row 4).
            loaded = []
            for fname in files:
                file_path = os.path.join(dataset.samples_dir, fname)
                if not os.path.exists(file_path):
                    raise FileNotFoundError(
                        f"Missing window {file_path} (bag {bag_id}) — dataset/index desynced")
                loaded.append(torch.load(file_path, weights_only=True, map_location='cpu').float())
            windows = torch.stack(loaded, dim=0)  # (W, T, C, H, Wg)
            if windows.dim() != 5:
                raise ValueError(
                    f"bag {bag_id}: expected 4D window tensors (T, C, H, Wg), "
                    f"got stacked shape {tuple(windows.shape)}")

            z = encode_windows(encoder, windows, device)  # (W, FEATURE_DIM), L2-normalized
            payload = make_bag_payload(z, bag_id=int(bag_id), subject=int(subject_idx),
                                       label=int(label_idx))
            torch.save(payload, os.path.join(out_dir, f"bag_{bag_id}.pt"))
            window_counts.add(int(payload['T_b']))

    # B1 canary: windows-per-bag must vary across trials; a single constant count
    # is the exact signature of the old time-axis swap.
    if len(window_counts) == 1 and len(dataset) > 10:
        raise RuntimeError(
            f"Every bag produced exactly {next(iter(window_counts))} windows — "
            "constant row count was the B1 corruption signature. Refusing to finish; "
            "investigate layouts before consuming these artifacts.")

    print(f"Precomputation complete: {len(dataset)} bags, "
          f"window counts min={min(window_counts)} max={max(window_counts)}")


def main():
    parser = argparse.ArgumentParser(description="Precompute Phase 1B bag artifacts (schema v2)")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to Phase 1A checkpoint')
    args = parser.parse_args()

    config = load_config(args.config)  # Hydra compose + architecture->model alias + seed
    run_precompute(config, args.checkpoint)


if __name__ == "__main__":
    main()
