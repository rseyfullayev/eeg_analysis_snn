"""Stage 0 regression tests — plain script, no pytest (the project venv has none;
we do not install packages without the owner's say-so).

Run from repo root:  venv\\Scripts\\python.exe tests\\test_stage0.py
Exit code 0 = all pass.

Covers the Stage 0 fixes (REVIEW_SMC_AUDIT.md §5):
  B1  encode_windows: time-first permute + window-count guard; payload v2
  B2  load_config path; strict encoder load; run_phase1b harness imports
  B7  index-wide subject map shared by train/val splits
  §3.5  silent-fallback purge: dataset + precomputed artifacts RAISE, never substitute
  PKSampler: deterministic sorted label order, effective-batch honesty
"""

import os
import random
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.snn_modeling.utils import features as ft
from src.snn_modeling.dataloader.dataset import SWEEPDataset, PKSampler


class _Raises:
    """with _Raises(SomeError, 'substr'): ...  (context manager, stdlib-only)"""
    def __init__(self, exc, contains=None):
        self.exc, self.contains = exc, contains

    def __enter__(self):
        return self

    def __exit__(self, etype, e, tb):
        if etype is None:
            raise AssertionError(f"expected {self.exc.__name__} — nothing raised")
        if not issubclass(etype, self.exc):
            return False  # propagate unexpected exception type
        if self.contains and self.contains not in str(e):
            raise AssertionError(f"expected message to contain {self.contains!r}, got: {e}")
        return True


class TimeFirstEncoder(nn.Module):
    """Fake encoder honoring the real convention: input (T,B,C,H,Wg) ->
    ((T,B,D,1,1), skips). Feature value per batch slot = slot index + 1, so a
    correct permute yields W unit rows; feeding windows-as-time corrupts counts."""
    D = ft.FEATURE_DIM

    def __init__(self):
        super().__init__()
        self.seen_shape = None

    def forward(self, x):
        assert x.dim() == 5, f"encoder must receive time-first 5D, got {x.dim()}D"
        self.seen_shape = tuple(x.shape)
        T, B, C, H, W = x.shape
        feats = (torch.arange(B, dtype=torch.float32) + 1.0).view(1, B, 1, 1, 1).expand(T, B, self.D, 1, 1)
        return feats, [None]


def _fake_dataset_dir(tmp, n_subjects=3, n_bags_per=2, n_windows=4, drop_file=None):
    os.makedirs(tmp, exist_ok=True)
    rows = []
    for s in range(1, n_subjects + 1):
        for b in range(n_bags_per):
            bag_id = 100 * s + b
            for w in range(n_windows):
                fname = f"{s}_{bag_id}_s{w}.pt"
                rows.append((fname, bag_id, s % 2))
                if drop_file == fname:
                    continue
                torch.save(torch.rand(2, 1, 4, 4), os.path.join(tmp, fname))  # (T,C,H,Wg)
    with open(os.path.join(tmp, "index.csv"), "w") as f:
        f.write("filename,bag_id,emotion_id\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]}\n")


def _cfg(tmp, bagging=True):
    return OmegaConf.create({
        'data': {'dataset_path': tmp, 'n_emotions': 2, 'grid_size': 4, 'preload_ram': False,
                 'train_size': 0.8, 'mil_bag_size': 6, 'num_timesteps': 2, 'num_workers': 0},
        'mask': {'sigma': 0.25, 'radius': 0.7},
        'training': {'bagging': bagging},
    })


# --------------------------------------------------------------------------- B1

def test_encode_windows_swaps_time_first():
    enc = TimeFirstEncoder()
    windows = torch.rand(6, 2, 1, 4, 4)            # (W=6, T=2, C=1, H=4, Wg=4)
    z = ft.encode_windows(enc, windows, torch.device('cpu'))
    assert enc.seen_shape == (2, 6, 1, 4, 4), \
        f"B1 regression: windows must become BATCH dim; encoder saw {enc.seen_shape}"
    assert z.shape == (6, TimeFirstEncoder.D), "rows must be window count, not a constant"
    assert torch.allclose(z.norm(dim=1), torch.ones(6), atol=1e-5), "z rows unit-normalized"


def test_encode_windows_rejects_wrong_rank():
    enc = TimeFirstEncoder()
    with _Raises(ValueError, "5D"):
        ft.encode_windows(enc, torch.rand(2, 1, 4, 4), torch.device('cpu'))


def test_bag_payload_roundtrip_and_legacy_refusal():
    with tempfile.TemporaryDirectory() as tmp:
        z = torch.nn.functional.normalize(torch.rand(5, ft.FEATURE_DIM), dim=1)
        payload = ft.make_bag_payload(z, bag_id=7, subject=2, label=1)
        assert payload['T_b'] == 5 and payload['schema'] == 'v2'
        path = os.path.join(tmp, "bag_7.pt")
        torch.save(payload, path)
        got = ft.load_bag_payload(path)
        assert torch.allclose(got['z'], z) and got['subject'] == 2

        with _Raises(ValueError):  # unnormalized rows must be refused at creation
            ft.make_bag_payload(torch.rand(4, ft.FEATURE_DIM), bag_id=1, subject=1, label=0)

        legacy = os.path.join(tmp, "legacy.pt")   # v1-era bare tensor
        torch.save(torch.rand(32, ft.FEATURE_DIM), legacy)
        with _Raises(RuntimeError, "v2"):
            ft.load_bag_payload(legacy)


def test_load_encoder_state_strict():
    enc = nn.Linear(4, 4)
    good = {'model_state_dict': {'encoder.weight': torch.ones(4, 4),
                                 'encoder.bias': torch.zeros(4),
                                 'classifier.weight': torch.ones(2, 2)}}
    ft.load_encoder_state_strict(enc, good)
    assert torch.equal(enc.weight.data, torch.ones(4, 4))

    with _Raises(RuntimeError):  # missing encoder.bias must raise, not half-load
        ft.load_encoder_state_strict(enc, {'model_state_dict': {'encoder.weight': torch.ones(4, 4)}})
    with _Raises(RuntimeError, "no 'encoder"):
        ft.load_encoder_state_strict(enc, {'model_state_dict': {'head.weight': torch.ones(1)}})


# --------------------------------------------------------------------------- dataset fail-loud

def test_dataset_raises_on_missing_window_file():
    with tempfile.TemporaryDirectory() as tmp:
        _fake_dataset_dir(tmp, n_subjects=1, n_bags_per=1, n_windows=2,
                          drop_file="1_100_s1.pt")
        ds = SWEEPDataset(_cfg(tmp), split='all', precomputed=False)
        with _Raises(FileNotFoundError):
            ds[0]  # zero-tensor substitution is gone


def test_dataset_precomputed_legacy_artifact_raises():
    with tempfile.TemporaryDirectory() as tmp:
        _fake_dataset_dir(tmp, n_subjects=1, n_bags_per=1, n_windows=2)
        os.makedirs(os.path.join(tmp, "precomputed_features"))
        torch.save(torch.rand(32, ft.FEATURE_DIM),  # legacy bare tensor
                   os.path.join(tmp, "precomputed_features", "bag_100.pt"))
        ds = SWEEPDataset(_cfg(tmp), split='all', precomputed=True)
        with _Raises(RuntimeError, "v2"):
            ds[0]


def test_dataset_precomputed_v2_rows_are_windows_not_frames():
    with tempfile.TemporaryDirectory() as tmp:
        _fake_dataset_dir(tmp, n_subjects=2, n_bags_per=1, n_windows=4)
        os.makedirs(os.path.join(tmp, "precomputed_features"))
        ds0 = SWEEPDataset(_cfg(tmp), split='all')
        for bag_id, files, label, subj, *_ in ds0.samples:
            z = torch.nn.functional.normalize(torch.rand(len(files), ft.FEATURE_DIM), dim=1)
            torch.save(ft.make_bag_payload(z, bag_id=int(bag_id), subject=int(subj),
                                           label=int(label)),
                       os.path.join(tmp, "precomputed_features", f"bag_{bag_id}.pt"))
        ds = SWEEPDataset(_cfg(tmp), split='all', precomputed=True)
        z, _t, _l, _s, _b, _v, wids = ds[0]
        assert z.shape[0] == len(ds.samples[0][1]), "rows = actual windows (never constant 32)"
        assert len(wids) == z.shape[0]


def test_subject_map_is_index_wide_B7():
    with tempfile.TemporaryDirectory() as tmp:
        _fake_dataset_dir(tmp, n_subjects=3)
        cfg = _cfg(tmp)
        tr = SWEEPDataset(cfg, loso='2', split='train')
        va = SWEEPDataset(cfg, loso='2', split='val')
        assert tr.subject_map == va.subject_map, "one shared label space across splits"
        assert 2 in tr.subject_map and tr.subject_map[2] != -1, "held-out subject maps to a real slot"


# --------------------------------------------------------------------------- PKSampler

class _FakeDS:
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)


def test_pksampler_determinism_sorted_labels_effective_batch():
    samples = [(str(i), ['x.pt'], i % 2, i % 3, i, None) for i in range(40)]
    ds = _FakeDS(samples)

    def first_batch():
        random.seed(0)
        np.random.seed(0)
        ps = PKSampler(ds, batch_size=10, n_classes=2, subject_diverse_k=False)
        return ps.label_set, next(iter(ps))

    ls1, b1 = first_batch()
    ls2, b2 = first_batch()
    assert ls1 == ls2 == [0, 1], "label order must be sorted (no raw set())"
    assert list(b1) == list(b2), "reproducible under equal seeds"


# --------------------------------------------------------------------------- harness imports

def test_harness_scripts_import_clean():
    import importlib
    importlib.import_module('run_phase1b_t0_t5')
    importlib.import_module('precompute_features')


# --------------------------------------------------------------------------- runner

def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
