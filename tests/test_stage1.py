"""Stage 1 regression tests — NullspaceState (space.py) per fixed.pdf §2/§6/§11.4.

Run from repo root:  venv\\Scripts\\python.exe tests\\test_stage1.py
Exit code 0 = all pass. (Plain asserts — the venv has no pytest.)

Covers the plan §5 Stage-1 checklist: P^2=P, Rm 6.5 exact unseen-subject
annihilation, norm floor to B^d, EMA/M-step geometry (§6.2-6.4), warmup
identity (§11.4(ii)), dynamic rank band + Rm 6.4 cap, strict lock roundtrip,
fail-loud inputs.

Synthetic geometry note: the four subject means below give eigen energy shares
[0.938, 0.062], so band 0.95 -> rank 2, band 0.5 -> rank 1 (tests use that).
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.snn_modeling.utils.space import NullspaceState, LOCK_SCHEMA

D, S = 8, 4
BAND2 = 0.95   # captures the 0.938/0.062 split -> exact rank 2


def _fit_state(**kw):
    """Four synthetic subjects whose (normalized) means lie in span{e1,e2}."""
    kw.setdefault("rank_energy_band", BAND2)
    st = NullspaceState(feature_dim=D, num_subjects=S,
                        warmup_epochs=kw.pop("warmup_epochs", 0), **kw)
    coords = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 1.0)]
    for s, (a, b) in enumerate(coords):
        mean = torch.zeros(D)
        mean[0], mean[1] = a, b
        z = F.normalize(mean.unsqueeze(0), dim=1)
        st.update_batch(z, torch.full((1,), 0.5), torch.tensor([s], dtype=torch.long))
    return st


def _basis_z():
    torch.manual_seed(7)
    return F.normalize(torch.rand(3, D), dim=1)


# ------------------------------------------------------------------ geometry

def test_projector_idempotent():
    st = _fit_state()
    assert int(st.rank.item()) == 2, f"expected band-exact rank 2, got {st.rank.item()}"
    h = torch.randn(5, D)
    p1 = st.apply_projector(h)
    p2 = st.apply_projector(p1)
    assert torch.allclose(p1, p2, atol=1e-6), "P(P h) != P(h)"


def test_unseen_subject_annihilation_rm65():
    st = _fit_state()
    r = int(st.rank.item())
    z = _basis_z()
    offset = (st.U[:, :r] @ torch.randn(r)).unsqueeze(0).expand_as(z)
    h1 = st.apply_projector(z - st.mu)
    h2 = st.apply_projector(z + offset - st.mu)
    assert torch.allclose(h1, h2, atol=1e-5), "unseen in-span offset survived the projector"


def test_norm_floor_ball():
    st = _fit_state()
    z = _basis_z()
    _, z_tilde, h_norm = st.project(z)
    norms = z_tilde.norm(dim=1)
    assert (norms <= 1.0 + 1e-6).all() and (norms >= 0.0).all(), "z_tilde left B^d"
    big = h_norm > st.tau_norm
    if big.any():
        assert torch.allclose(norms[big], torch.ones(int(big.sum())), atol=1e-5), \
            "rows with ||h'|| >= tau must land on the sphere"
    # a row whose h' is inside span(U) floors to exactly zero
    in_span = F.normalize(st.U[:, :1], dim=0).t() + st.mu  # mu + in-span dir
    in_span = F.normalize(in_span, dim=1)
    _, t_zero, h_zero = st.project(in_span)
    assert float(h_zero[0]) < 1e-4 and float(t_zero.norm()) < 1e-4, "in-span residual not annihilated"


def test_warmup_identity_then_active():
    st = _fit_state(warmup_epochs=2)
    assert int(st.rank.item()) >= 1, "geometry must keep computing during warmup (only projection is off)"
    h = torch.randn(3, D)
    assert torch.equal(st.apply_projector(h), h), "warmup must not project"
    st.advance_epoch()
    assert torch.equal(st.apply_projector(h), h), "epoch 1 < 2 still warmup"
    st.advance_epoch()
    p = st.apply_projector(h)
    assert not torch.allclose(p, h, atol=1e-4), "post-warmup projection became a no-op"


# ------------------------------------------------------------------ M-steps

def test_ema_means_and_global_mu():
    st = NullspaceState(feature_dim=D, num_subjects=S, ema_momentum=0.0,
                        rank_energy_band=BAND2, warmup_epochs=0)
    m1 = torch.zeros(D); m1[0] = 1.0
    m2 = torch.zeros(D); m2[1] = 1.0
    z = torch.stack([m1, m2])
    alpha = torch.tensor([0.25, 0.75])          # baseline weights w = 0.75, 0.25
    subj = torch.tensor([0, 1], dtype=torch.long)
    st.update_batch(z, alpha, subj)
    assert torch.allclose(st.ema_mean[0], m1, atol=1e-6)
    assert abs(float(st.ema_mass[0]) - 0.75) < 1e-6 and abs(float(st.ema_mass[1]) - 0.25) < 1e-6
    expected_mu = 0.75 * m1 + 0.25 * m2          # §6.4 mass-weighted global mean (total mass 1)
    assert torch.allclose(st.mu, expected_mu, atol=1e-6)


def test_rank_cap_and_monotone_band():
    lo = _fit_state(rank_energy_band=0.5)
    hi = _fit_state(rank_energy_band=0.99)
    assert int(lo.rank.item()) == 1 and int(hi.rank.item()) == 2, \
        f"band monotonicity broken: {lo.rank.item()} vs {hi.rank.item()}"
    st = NullspaceState(feature_dim=D, num_subjects=S, rank_energy_band=0.99, warmup_epochs=0)
    torch.manual_seed(3)
    for s in range(S):
        v = torch.zeros(1, D); v[0, s] = 1.0     # near-orthogonal means -> raw span S, cap S-1
        st.update_batch(F.normalize(v + 0.01 * torch.randn(1, D), dim=1),
                        torch.full((1,), 0.5), torch.tensor([s], dtype=torch.long))
    assert int(st.rank.item()) <= S - 1, "Rm 6.4: rank(S_between) <= S-1 violated"


def test_no_data_yields_identity_state():
    st = NullspaceState(feature_dim=D, num_subjects=S, warmup_epochs=0)
    z = _basis_z()
    h, t, _ = st.project(z)
    assert torch.allclose(h, z, atol=1e-6) and torch.allclose(t, z, atol=1e-3), \
        "empty state must be identity-project with floor only"


# ------------------------------------------------------------------ lock I/O

def test_lock_roundtrip_and_rejections():
    st = _fit_state()
    payload = st.lock_payload()
    r = payload["rank"]
    assert payload["schema"] == LOCK_SCHEMA and payload["U"].shape == (D, r) and r == 2

    st2 = NullspaceState(feature_dim=D, num_subjects=S, rank_energy_band=BAND2, warmup_epochs=0)
    st2.load_lock(payload)
    z = _basis_z()
    assert torch.allclose(st.project(z)[1], st2.project(z)[1], atol=1e-6)

    def _expect_raise(mut):
        bad = dict(payload)
        bad.update(mut)
        try:
            st2.load_lock(bad)
        except ValueError:
            return
        raise AssertionError(f"corruption accepted: {mut}")

    _expect_raise({"schema": "smc-space-v0"})
    _expect_raise({"rank": S})                       # > max_rank, shape mismatch
    _expect_raise({"U": payload["U"] * 2.0})        # non-orthonormal columns
    _expect_raise({"mu": torch.full((D,), float("nan"))})
    _expect_raise({"feature_dim": D + 1})


# ------------------------------------------------------------------ fail-loud input

def test_inputs_fail_loud():
    st = _fit_state()
    z = _basis_z()

    def case(fn):
        return fn

    cases = [
        lambda: st.project(z * 2.0),                                             # non-unit rows
        lambda: st.project(torch.randn(3, D + 1)),                               # wrong dim
        lambda: st.update_batch(z, torch.full((3,), 1.5), torch.tensor([0, 1, 2])),   # alpha>1
        lambda: st.update_batch(z, torch.full((3,), 0.5), torch.tensor([0, 1, 99])),  # subject OOR
        lambda: st.update_batch(z, torch.full((3,), 0.5), torch.tensor([-1, 0, 1])),  # -1 sentinel
        lambda: st.update_batch(z, torch.full((3,), 0.5), torch.tensor([0., 1., 2.])),  # wrong dtype
        lambda: st.update_batch(z * 2.0, torch.full((3,), 0.5), torch.tensor([0, 1, 2])),  # non-unit z
    ]
    for i, fn in enumerate(cases):
        try:
            fn()
            raise AssertionError(f"case {i}: bad input accepted silently")
        except ValueError:
            pass  # expected


# ------------------------------------------------------------------ runner

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


if __name__ == "__main__":
    sys.exit(main())
