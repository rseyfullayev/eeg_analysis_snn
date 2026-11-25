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
    from src.snn_modeling.layers.stem import MeanMaxPoolHead
    enc = TimeFirstEncoder()
    pool = MeanMaxPoolHead(TimeFirstEncoder.D, kernel=1)   # stub emits (T,B,D,1,1)
    windows = torch.rand(6, 2, 1, 4, 4)            # (W=6, T=2, C=1, H=4, Wg=4)
    z = ft.encode_windows(enc, windows, torch.device('cpu'), pool_head=pool)
    assert enc.seen_shape == (2, 6, 1, 4, 4), \
        f"B1 regression: windows must become BATCH dim; encoder saw {enc.seen_shape}"
    assert z.shape == (6, TimeFirstEncoder.D), "rows must be window count, not a constant"
    assert torch.allclose(z.norm(dim=1), torch.ones(6), atol=1e-5), "z rows unit-normalized"


def test_encode_windows_requires_shared_pool_head():
    # The trained embedding is encoder+head together: encode_windows must not
    # offer a private mean-GAP fallback (B1-class train/serve skew).
    import inspect
    sig = inspect.signature(ft.encode_windows).parameters
    assert 'pool_head' in sig and sig['pool_head'].default is inspect.Parameter.empty, \
        "pool_head must be a required keyword (no silent GAP default)"


def test_mean_max_pool_head_semantics():
    """Owner redesign 2026-11-20: depthwise 4x4 pad-0 squeeze on the 4x4
    stage-4 map, per-channel 0.5*mean_t + 0.5*max_t, no activation pre-pool."""
    from src.snn_modeling.layers.stem import MeanMaxPoolHead
    head = MeanMaxPoolHead(256, kernel=4)
    head.eval()
    with torch.no_grad():
        head.dw.weight.fill_(0.0)
        head.dw.weight[:, 0, :, :] = 1.0 / 16.0     # every channel = mean of its 4x4 field
        head.norm.weight.fill_(1.0); head.norm.bias.fill_(0.0)
        m = torch.zeros(8, 1, 256, 4, 4)
        m[3, 0, 5] += 1000.0                        # one-frame burst in channel 5
        per_frame = head.norm(head.dw(m.reshape(8, 256, 4, 4))).view(8, 1, 256)
        expected = 0.5 * per_frame.mean(dim=0) + 0.5 * per_frame.max(dim=0).values
        out = head(m)
    assert torch.allclose(out, expected, atol=1e-5), "blend must be exactly 0.5*mean+0.5*max"
    z_peak = float(per_frame[3, 0, 5])
    assert z_peak > 0 and float(out[0, 5]) > 0.45 * z_peak, \
        "max term must retain the burst peak (mean-only would give z_peak/8)"
    with _Raises(ValueError, "time-first"):
        head(torch.zeros(256, 4, 4))                # 3D (no T/B axes) must fail loudly
    with _Raises(ValueError, "HxW == kernel"):
        head(torch.zeros(8, 1, 256, 2, 2))          # stale 2x2 schedule must fail loudly
    # NOTE: a batch-first (B,T,C,H,W) map is shape-indistinguishable here; T/B
    # swap is caught upstream by encode_windows' window-count guard (B1 test).
    xin = torch.randn(4, 2, 256, 4, 4, requires_grad=True)
    head(xin).sum().backward()
    assert torch.isfinite(xin.grad).all() and head.dw.weight.grad is not None, \
        "grads must flow through squeeze+GN+pool"


def test_real_projector_end_to_end_4x4_schedule():
    """Backbone schedule pinned: stage4 stays 4x4 (stride 1, dilation 2) and
    projector.extract_features returns (B, FEATURE_DIM) through the shared head."""
    from src.snn_modeling.models.encoders import SpikingMobileNetEncoder
    from src.snn_modeling.models.unet import SpikingMobileNetProjector
    enc = SpikingMobileNetEncoder(in_channels=5, p_drop=0.3, use_odconv=True,
                                  use_batchnorm=False, spike_model="SiLU")
    model = SpikingMobileNetProjector(encoder_backbone=enc, num_classes=5)
    model.eval()
    x = torch.randn(8, 2, 5, 32, 32)               # (T,B,C,H,Wg), small batch
    with torch.no_grad():
        s = model.encoder.stem(x)
        s = model.encoder.stage1(s); s = model.encoder.stage2(s)
        s = model.encoder.stage3(s); s4 = model.encoder.stage4(s)
        assert tuple(s4.shape[-2:]) == (4, 4), f"stage4 must stay 4x4, got {tuple(s4.shape)}"
        feats = model.extract_features(x)
    assert feats.shape == (2, ft.FEATURE_DIM)
    assert torch.isfinite(feats).all()


def test_encode_windows_rejects_wrong_rank():
    from src.snn_modeling.layers.stem import MeanMaxPoolHead
    enc = TimeFirstEncoder()
    with _Raises(ValueError, "5D"):
        ft.encode_windows(enc, torch.rand(2, 1, 4, 4), torch.device('cpu'),
                          pool_head=MeanMaxPoolHead(TimeFirstEncoder.D, kernel=1))


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

def test_dataset_trains_without_masks_pt():
    """2026-11-25: masks.pt is OPTIONAL — no SMC Phase-1A consumer reads the target
    maps. With prototypes=None the dataset keeps its documented analytic default and
    the batch tuple/collate shape is unchanged (target_map slot still (grid,grid) long)."""
    with tempfile.TemporaryDirectory() as tmp:
        _fake_dataset_dir(tmp, n_subjects=1, n_bags_per=1, n_windows=2)
        assert not os.path.exists(os.path.join(tmp, "masks.pt"))
        ds = SWEEPDataset(_cfg(tmp), split='all', precomputed=False, prototypes=None)
        item = ds[0]
        tmap = item[1]
        assert tuple(tmap.shape) == (ds.grid_size, ds.grid_size) and tmap.dtype == torch.long
        assert int(tmap.max()) >= 1  # analytic prototypes still produce a foreground map
        # a mask-built dataset with the SAME index must differ only in the map contents
        m = torch.zeros(ds.num_classes, ds.grid_size, ds.grid_size)
        m[0, 0, 0] = 1.0
        ds_m = SWEEPDataset(_cfg(tmp), split='all', precomputed=False, prototypes=m)
        assert int(ds_m[0][1].max()) in (0, 1)  # no crash when a class maps to nothing
        # default-collate contract survives (train.py uses the stock collater)
        from torch.utils.data import default_collate
        b = default_collate([ds[0], ds[0]])
        assert b[1].shape[1:] == (ds.grid_size, ds.grid_size) and b[1].dtype == torch.long
        # window_ids slot must be a LONG TENSOR (pod crash 2026-11-26): default_collate
        # turns per-sample Python LISTS into a list of per-position tensors (never a
        # stacked tensor), which killed train.py's `timestamps.to(device)`.
        assert isinstance(item[5], torch.Tensor) and item[5].dtype == torch.long
        assert isinstance(b[5], torch.Tensor) and b[5].dtype == torch.long


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


# --------------------------------------------------------------------------- asymmetric dual view

class _Doubles(__import__('torch').nn.Module):
    """Deterministic marker: any chain containing it is visibly not the clean view."""

    def forward(self, x):
        return x * 2.0


def _aug_cfg(tmp):
    import torch.nn as nn
    return nn.Sequential(_Doubles())


def test_key_view_must_be_declared():
    """One chain for BOTH views is the donor-leak defect: an active query chain with an
    undeclared key chain must raise, not silently reuse the query chain."""
    with tempfile.TemporaryDirectory() as tmp:
        _fake_dataset_dir(tmp)
        ds = SWEEPDataset(_cfg(tmp), loso='3', split='train', augmentations=_aug_cfg(tmp))
        try:
            ds[0]
        except RuntimeError as e:
            assert 'key_augmentations' in str(e), f"wrong guard message: {e}"
        else:
            raise AssertionError("dual-view dataset accepted an undeclared key view")


def test_key_view_clean_is_unaugmented():
    import torch
    with tempfile.TemporaryDirectory() as tmp:
        _fake_dataset_dir(tmp)
        clean = SWEEPDataset(_cfg(tmp), loso='3', split='train')
        both = SWEEPDataset(_cfg(tmp), loso='3', split='train',
                            augmentations=_aug_cfg(tmp), key_augmentations=None)
        v1, v2 = both[0][0], both[0][1]
        ref = clean[0][0]
        assert torch.equal(v2, ref), "clean key view must be the untouched window"
        assert torch.equal(v1, ref * 2.0), "query view must still be augmented"


def test_key_view_rejects_fourier_mixup():
    import torch.nn as nn
    from src.snn_modeling.utils.augmentations import FourierMixup
    with tempfile.TemporaryDirectory() as tmp:
        _fake_dataset_dir(tmp)
        # nested inside a Sequential — a top-level isinstance check would miss it
        ds = SWEEPDataset(_cfg(tmp), loso='3', split='train',
                          augmentations=_aug_cfg(tmp),
                          key_augmentations=nn.Sequential(FourierMixup(sigma=2.0, shape=(4, 4, 4)),
                                                          _Doubles()))
        try:
            ds[0]
        except RuntimeError as e:
            assert 'FourierMixup' in str(e), f"wrong guard message: {e}"
        else:
            raise AssertionError("key view accepted a chain containing FourierMixup")


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


def test_sklearn_probe_factories_fit():
    """The Table-1 probes had NO local coverage until 2026-11-26: get_logistic_regression
    passed multi_class='multinomial', REMOVED in the pinned scikit-learn 1.7.2, every
    fold raised TypeError, and the pod printed 0.0000 for all LOTO rows. These factories
    must construct, fit and predict on THIS interpreter's sklearn (same pinned version
    as the pod image install)."""
    import numpy as np
    from test import get_logistic_regression, get_svm
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(m, 1.0, (150, 8)) for m in range(3)])
    y = np.repeat([0, 1, 2], 150)
    lr = get_logistic_regression()
    lr.fit(X, y)
    assert (lr.predict(X) == y).mean() > 0.8, "LogReg probe stopped discriminating"
    svc = get_svm()
    svc.fit(X, y)
    assert (svc.predict(X) == y).mean() > 0.8, "LinearSVC probe stopped discriminating"


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
