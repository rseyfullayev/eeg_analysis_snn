"""Stage 2 regression tests — Queue v2 (memory/queue.py) per fixed.pdf §9.

Run from repo root:  venv\\Scripts\\python.exe tests\\test_stage2.py
Exit code 0 = all pass. (Plain asserts — the venv has no pytest.)

Covers plan §5 Stage-2 checklist: ring wrap-around with all five fields,
§9.4 window-id masking on BOTH readouts, beta-weighted class clouds (Rm 9.2),
Prop 9.2 LLN smoke on a synthetic 2-component mixture, ESS_y closed form,
§9.6 sizing + §9.4 staleness formulas, and fail-loud enqueue validation.
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.snn_modeling.memory.queue import (SupMoCoQueue, check_sizing, check_staleness,
                                           max_staleness, queue_ceiling, queue_floor)

D = 6


def _q(size=8, C=3):
    return SupMoCoQueue(size, D, num_classes=C)


def _keys(n, seed=0):
    torch.manual_seed(seed)
    return F.normalize(torch.randn(n, D), dim=1)


# ------------------------------------------------------------------ validation

def test_enqueue_fail_loud():
    q = _q()
    k = _keys(4)
    lbl = torch.tensor([0, 1, 2, 0])
    b = torch.full((4,), 0.3)
    w = torch.arange(4, dtype=torch.long)
    for fn in [
        lambda: q.enqueue(k * 2.0, lbl, b, w, 0),                       # outside B^d
        lambda: q.enqueue(k, torch.tensor([0, 1, 7, 0]), b, w, 0),      # label >= C
        lambda: q.enqueue(k, lbl, torch.tensor([0.3, 1.5, 0.2, 0.1]), w, 0),   # beta > 1
        lambda: q.enqueue(k, lbl, b, torch.tensor([1, 2, 3, 3]), 0),    # dup window ids
        lambda: q.enqueue(k, lbl, b, w.float(), 0),                     # wrong dtype
        lambda: q.enqueue(k, lbl, b, w, -1),                            # negative step
        lambda: q.enqueue(k.detach().requires_grad_(), lbl, b, w, 0),   # grad-tracking keys
    ]:
        try:
            fn()
            raise AssertionError("bad input accepted silently")
        except (ValueError, RuntimeError):
            pass
    q.enqueue(k.detach(), lbl, b, w, 0)  # control: clean call works


def test_grad_is_rejected_not_silently_detached():
    q = _q()
    k = _keys(2).requires_grad_(True)
    try:
        q.enqueue(k, torch.tensor([0, 1]), torch.tensor([0.5, 0.5]),
                  torch.tensor([10, 11]), 0)
        raise AssertionError("grad-tracked keys were accepted")
    except RuntimeError:
        pass


# ------------------------------------------------------------------ ring mechanics

def test_ring_wraparound_and_full_flush():
    q = _q(size=8)
    for i in range(3):                       # 4+4+4 = 12 into size 8
        q.enqueue(_keys(4, seed=i), torch.full((4,), i % 3), torch.full((4,), 0.25),
                  torch.arange(i * 4, i * 4 + 4, dtype=torch.long), i)
    assert int(q.filled.item()) == 8
    k, lbl, b, w = q.read_negatives()
    assert k.shape == (8, D) and w.numel() == 8
    # the first batch (ids 0-3) was fully overwritten by the third (ids 8-11)
    ids = set(w.tolist())
    assert {8, 9, 10, 11} <= ids and not ({0, 1, 2, 3} & ids), f"ring eviction wrong: {sorted(ids)}"
    # n >= queue_size path: one huge batch flushes the queue, keeps the tail
    q.enqueue(_keys(10, seed=9), torch.zeros(10, dtype=torch.long), torch.full((10,), 0.1),
              torch.arange(100, 110, dtype=torch.long), 9)
    _, _, _, w2 = q.read_negatives()
    assert set(w2.tolist()) == set(range(102, 110)), "oversized enqueue must keep last queue_size entries"


def test_window_id_masking_both_readouts():
    q = _q(size=16)
    q.enqueue(_keys(8, seed=1), torch.tensor([0, 1] * 4), torch.full((8,), 0.5),
              torch.arange(8, dtype=torch.long), 0)
    excl = torch.tensor([2, 5])
    _, _, _, wid = q.read_negatives(exclude_window_ids=excl)
    assert set(wid.tolist()) & {2, 5} == set(), "leakage guard failed on negatives"
    clouds = q.read_class_clouds(exclude_window_ids=excl)
    all_ids = set()
    for y, c in clouds.items():
        # recompute ids via masking indirectly: counts must drop exactly where excluded
        all_ids.add(y)
    assert 0 in all_ids and 1 in all_ids
    n_before = sum(c["n"] for c in q.read_class_clouds().values())
    n_after = sum(c["n"] for c in clouds.values())
    assert n_before - n_after == 2, f"cloud masking dropped {n_before - n_after} != 2 entries"


def test_cloud_weights_and_zero_beta():
    q = _q(size=4)
    q.enqueue(_keys(3, seed=2), torch.tensor([0, 0, 1]),
              torch.tensor([0.5, 0.5, 0.25]), torch.arange(3, dtype=torch.long), 0)
    clouds = q.read_class_clouds()
    c0 = clouds[0]
    assert c0["n"] == 2 and abs(c0["weights"].sum().item() - 1.0) < 1e-6
    assert abs(c0["beta_sum"] - 1.0) < 1e-6
    assert abs(clouds[1]["weights"].item() - 1.0) < 1e-6   # single-entry cloud weights to 1
    # a cloud whose every entry has beta=0 must raise (interiority violation)
    q2 = _q(size=2)
    q2.enqueue(_keys(1, seed=3), torch.tensor([1]), torch.tensor([0.0]),
               torch.tensor([7]), 0)
    try:
        q2.read_class_clouds()
        raise AssertionError("zero-sum beta cloud accepted")
    except ValueError:
        pass


def test_prop92_lln_smoke():
    """beta-weighted class slice converges to the burst component (Prop 9.2).
    Points live inside B^2 (norms < 1) as the queue requires."""
    q = SupMoCoQueue(4000, 2, num_classes=1)
    torch.manual_seed(0)
    rho = 0.15
    n = 3000
    mu_b = torch.tensor([0.5, 0.0])
    burst = torch.randn(n, 2) * 0.03 + mu_b
    base = torch.randn(n, 2) * 0.05
    is_b = torch.rand(n) < rho
    pts = torch.where(is_b.unsqueeze(1), burst, base)
    assert pts.norm(dim=1).max() < 1.0, "synthetic keys must live in B^d"
    # exact posterior weights for the 2-component mixture
    lb = -((pts - mu_b) ** 2).sum(1) / (2 * 0.03 ** 2)
    l0 = -(pts ** 2).sum(1) / (2 * 0.05 ** 2)
    beta = torch.sigmoid(lb - l0 + torch.log(torch.tensor(rho / (1 - rho))))
    q.enqueue(pts, torch.zeros(n, dtype=torch.long), beta,
              torch.arange(n, dtype=torch.long), 0)
    clouds = q.read_class_clouds()
    wsum = (clouds[0]["keys"] * clouds[0]["weights"].unsqueeze(1)).sum(0)
    assert (wsum - mu_b).norm() < 0.05, f"weighted mean off burst center: {wsum}"
    unweighted = clouds[0]["keys"].mean(0)
    assert unweighted[0] < 0.25 and wsum[0] > 0.45, \
        f"unweighted ({unweighted}) should sit near origin, weighted ({wsum}) at burst"


def test_ess_closed_form():
    q = _q(size=8)
    q.enqueue(_keys(3, seed=4), torch.tensor([0, 0, 0]),
              torch.tensor([0.5, 0.25, 0.25]), torch.arange(3, dtype=torch.long), 0)
    ess = q.ess_by_class()
    expected = (1.0 ** 2) / (0.5 ** 2 + 0.25 ** 2 + 0.25 ** 2)
    assert abs(ess[0] - expected) < 1e-6, f"ESS {ess[0]} != {expected}"
    ess2 = q.ess_by_class(exclude_window_ids=torch.tensor([0]))
    assert abs(ess2[0] - (0.5 ** 2) / (0.25 ** 2 + 0.25 ** 2)) < 1e-6


# ------------------------------------------------------------------ §9.6 / §9.4 formulas

def test_sizing_and_staleness_formulas():
    fl = queue_floor(C_const=4.0, d_eff=64.0, tol=0.1, rho_tilde_min=0.05)
    assert fl == int(4 * 64 / (0.01 * 0.05))
    ce = queue_ceiling(eps_tol=0.05, B=64, T_total=12000, L=8.0, D=300)
    assert queue_ceiling(eps_tol=0.05, B=32, T_total=12000, L=8.0, D=300) == ce // 2  # linear in B
    check_sizing(512, floor=500, ceiling=900)
    for bad in [(512, 600, 900), (512, 100, 500)]:
        try:
            check_sizing(*bad)
            raise AssertionError(f"sizing {bad} accepted")
        except ValueError:
            pass
    st = max_staleness(n_q=512, B=64, momentum=0.9, L=2.0, D=100)
    assert abs(st - 2.0 * 8 * 0.1 * 100) < 1e-9
    try:
        check_staleness(n_q=512, B=64, momentum=0.0, L=2.0, D=100, eps_tol=10.0)
        raise AssertionError("staleness violation accepted")
    except ValueError:
        pass
    try:
        queue_floor(4.0, 64.0, 0.0, 0.05)   # tol must be positive
        raise AssertionError("tol=0 accepted")
    except ValueError:
        pass


# ------------------------------------------------------------------ summary

def test_summary_empty_and_full():
    q = _q(size=4)
    s = q.summary()
    assert s["filled"] == 0 and s["oldest_step"] == -1
    q.enqueue(_keys(2, seed=5), torch.tensor([0, 1]), torch.tensor([0.4, 0.6]),
              torch.tensor([3, 4]), step=7)
    s = q.summary()
    assert s["filled"] == 2 and s["oldest_step"] == 7 and abs(s["fill_ratio"] - 0.5) < 1e-9


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
