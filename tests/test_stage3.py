"""Stage 3 regression tests — estep.py + sparsity.py (fixed.pdf Eq. 6/12/14-17, §9.5, Rm 4.2).

Run from repo root:  venv\\Scripts\\python.exe tests\\test_stage3.py
Exit code 0 = all pass. (Plain asserts — the venv has no pytest.)

Covers plan §5 Stage-3 checklist: seed-loop contraction (B6(v)); T->inf =>
alpha ~= rho_y (cold-queue bootstrap); bkl minimiser at a=rho and f''>=4;
G^2 two-step baseline with error bound (16) on synthetic bags; alpha^(1B)
invariant under class permutation (label-free by construction); LSE kernel
== Sinkhorn cost matrix (Rm 4.2 sharing); every fail-loud gate.
"""

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.snn_modeling.utils.estep import (alpha_1a, alpha_1b, anneal_temperature,
                                          bag_statistic, batch_two_step_baselines,
                                          cloud_log_p_matrix, log_baseline_gaussian,
                                          two_step_baseline)
from src.snn_modeling.utils.sparsity import (bkl, burst_delta_scores, option_b_seed,
                                             rho_mstep, sparsity_loss)

D = 8


def _expect(exc, fn, msg):
    try:
        fn()
    except exc:
        return
    raise AssertionError(msg)


# ------------------------------------------------------------------ §9.5 seeding

def test_seed_fixed_point_and_contraction():
    # symmetric scores -> fixed point u*=0, rho=0.5, contraction reached in one step
    rho, diag = option_b_seed(torch.tensor([-2.0, -1.0, 1.0, 2.0]), rho_min=0.02)
    assert abs(rho - 0.5) < 1e-5 and diag["iterations"] == 1
    assert diag["kappa"] < 1.0 and not diag["saturated"]
    # asymmetric -> nontrivial interior fixed point (B6(v) iteration converges)
    rho, diag = option_b_seed(torch.tensor([-4.0, -1.0, 1.0, 2.0]), rho_min=0.02)
    u = diag["u_star"]
    d64 = torch.tensor([-4.0, -1.0, 1.0, 2.0], dtype=torch.float64)
    assert abs(torch.sigmoid(d64 + u).mean().item() - 1.0 / (1.0 + math.exp(-u))) < 1e-8, \
        "not a fixed point of F"
    assert diag["kappa"] < 1.0 and 0.02 < rho < 0.98 and not diag["saturated"]
    assert set(diag) == {"iterations", "residual", "kappa", "u_star", "saturated"}
    # saturation = the Eq. 12 clamp landing on the interiority box boundary
    rho_lo, d_lo = option_b_seed(-50.0 + 0.01 * torch.randn(64), rho_min=0.02)
    assert d_lo["saturated"] and abs(rho_lo - 0.02) < 1e-6
    rho_hi, d_hi = option_b_seed(50.0 + 0.01 * torch.randn(64), rho_min=0.02)
    assert d_hi["saturated"] and abs(rho_hi - 0.98) < 1e-6
    # degenerate scores: Var(P)=0 -> F'=1, contraction hypothesis violated (B6(v))
    _expect(ValueError, lambda: option_b_seed(torch.full((32,), 1.7), rho_min=0.02),
            "degenerate seed accepted")
    _expect(RuntimeError, lambda: option_b_seed(torch.tensor([-4.0, -1.0, 1.0, 2.0]),
                                                rho_min=0.02, max_iter=2, tol=1e-30),
            "non-convergent seed accepted")
    for bad in [lambda: option_b_seed(torch.empty(0), 0.02),
                lambda: option_b_seed(torch.tensor([0.0, float("nan")]), 0.02),
                lambda: option_b_seed(torch.randn(8), 0.0),
                lambda: option_b_seed(torch.randn(8), 0.5),
                lambda: option_b_seed(torch.randn(8), 0.02, max_iter=0)]:
        _expect(ValueError, bad, "bad seed input accepted")


def test_burst_delta_scores_formula():
    d2 = torch.tensor([0.0, 1.0, 4.0])
    out = burst_delta_scores(d2, s=2.0, log_coeff=6.0)
    expect = 0.5 * (1.0 - 0.25) * d2 - 6.0 * math.log(2.0)
    assert torch.allclose(out, expect, atol=1e-6), "Delta != ((1-s^-2)/2)||r||^2 - B log s"
    for bad in [lambda: burst_delta_scores(d2, s=1.0, log_coeff=6.0),
                lambda: burst_delta_scores(d2, s=float("nan"), log_coeff=6.0),
                lambda: burst_delta_scores(d2, s=2.0, log_coeff=0.0),
                lambda: burst_delta_scores(torch.tensor([-0.5]), s=2.0, log_coeff=1.0),
                lambda: burst_delta_scores(torch.tensor([float("inf")]), s=2.0, log_coeff=1.0)]:
        _expect(ValueError, bad, "bad delta input accepted")


def test_anneal_temperature():
    total = 1000
    assert anneal_temperature(0, total) == 20.0
    assert abs(anneal_temperature(150, total) - 10.5) < 1e-9
    assert anneal_temperature(300, total) == 1.0 and anneal_temperature(999, total) == 1.0
    ts = [anneal_temperature(s, total) for s in range(0, total, 37)]
    assert all(a >= b for a, b in zip(ts, ts[1:])), "annealing must be non-increasing"
    for bad in [lambda: anneal_temperature(0, 0),
                lambda: anneal_temperature(-1, 10),
                lambda: anneal_temperature(0, 10, t_start=1.0, t_end=20.0),
                lambda: anneal_temperature(0, 10, anneal_frac=0.0),
                lambda: anneal_temperature(0, 10, anneal_frac=1.5)]:
        _expect(ValueError, bad, "bad schedule input accepted")


# ------------------------------------------------------------------ bkl / Eq. 12 / Eq. 21

def test_bkl_prop121():
    rho = torch.tensor(0.3, dtype=torch.float64)
    a = torch.linspace(0.0, 1.0, 201, dtype=torch.float64)
    v = bkl(a, rho)
    assert (v >= -1e-12).all(), "bkl negative somewhere (Prop 12.1(i))"
    assert abs(float(v[60])) < 1e-12, "bkl(a, rho) != 0 at a = rho"
    assert int(v.argmin()) == 60, "minimiser not at rho"
    # endpoints via the continuous extension (Prop 12.1(ii))
    assert abs(float(bkl(torch.tensor([0.0], dtype=torch.float64), rho))
               + math.log(1.0 - 0.3)) < 1e-12
    assert abs(float(bkl(torch.tensor([1.0], dtype=torch.float64), rho))
               + math.log(0.3)) < 1e-12
    # strict convexity: f''(a) = 1/(a(1-a)) >= 4 (B6(i))
    for a0 in (0.05, 0.3, 0.5, 0.7, 0.95):
        h = 1e-4
        at = torch.tensor([a0], dtype=torch.float64)
        d2 = (float(bkl(at + h, rho)) - 2 * float(bkl(at, rho)) + float(bkl(at - h, rho))) / h ** 2
        assert abs(d2 - 1.0 / (a0 * (1 - a0))) < 1.0, f"f'' mismatch at a={a0}"
        assert d2 >= 4.0 - 0.05, f"f'' < 4 violated at a={a0}"
    # analytic gradient  d/da bkl = logit(a) - logit(rho)
    ag = torch.tensor([0.4, 0.6], dtype=torch.float64, requires_grad=True)
    bkl(ag, torch.tensor([0.25, 0.75], dtype=torch.float64)).sum().backward()
    expect = torch.log(ag.detach() / (1 - ag.detach())) - \
        torch.log(torch.tensor([0.25, 0.75], dtype=torch.float64) /
                  torch.tensor([0.75, 0.25], dtype=torch.float64))
    assert torch.allclose(ag.grad, expect, rtol=1e-6), "autograd gradient != closed form"
    _expect(ValueError, lambda: bkl(torch.tensor([0.5]), torch.tensor([0.3]).requires_grad_()),
            "grad-tracking rho accepted (must be detached M-step quantity)")
    for bad in [lambda: bkl(torch.tensor([1.5]), torch.tensor([0.3])),
                lambda: bkl(torch.tensor([0.5]), torch.tensor([0.0])),
                lambda: bkl(torch.tensor([0.5]), torch.tensor([1.0])),
                lambda: bkl(torch.tensor([float("nan")]), torch.tensor([0.3]))]:
        _expect(ValueError, bad, "bad bkl input accepted")


def test_rho_mstep_and_sparsity_loss():
    alpha = torch.tensor([0.2, 0.4, 0.8, 0.9, 0.5], requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
    rho = rho_mstep(alpha, labels, num_classes=3, rho_min=0.02)
    assert torch.allclose(rho, torch.tensor([0.3, 0.85, 0.5]), atol=1e-6), "Eq. 12 mean wrong"
    assert not rho.requires_grad and rho.grad_fn is None, "Eq. 12 must be detached"
    # clamp at both ends
    ext = rho_mstep(torch.tensor([0.001, 0.001, 0.999, 0.999]),
                    torch.tensor([0, 0, 1, 1], dtype=torch.long), 2, rho_min=0.01)
    assert abs(float(ext[0]) - 0.01) < 1e-6 and abs(float(ext[1]) - 0.99) < 1e-6
    # missing class: no prev -> raise; explicit prev -> carried
    _expect(ValueError, lambda: rho_mstep(alpha[:2], labels[:2], 3, 0.02),
            "silent rho fallback allowed")
    prev = torch.tensor([0.3, 0.85, 0.4])
    sub = rho_mstep(alpha[:2], labels[:2], 3, 0.02, prev_rho=prev)
    assert abs(float(sub[2]) - 0.4) < 1e-6, "prev_rho not carried"
    # L_spr = sum_t bkl(alpha_t, rho_{y_t}) — manual
    loss = sparsity_loss(alpha.detach(), labels, rho)
    manual = sum(float(bkl(alpha[i].detach(), rho[labels[i]])) for i in range(5))
    assert abs(float(loss) - manual) < 1e-6, "Eq. 21 sum mismatch"
    ag = torch.tensor([0.4, 0.2], requires_grad=True)
    l2 = sparsity_loss(ag, torch.tensor([0, 1], dtype=torch.long),
                       torch.tensor([0.7, 0.1]))
    l2.backward()
    assert ag.grad is not None and torch.isfinite(ag.grad).all(), "L_spr must be differentiable in alpha"
    for bad in [lambda: sparsity_loss(torch.empty(0), torch.empty(0, dtype=torch.long), rho),
                lambda: sparsity_loss(alpha, labels.float(), rho),
                lambda: sparsity_loss(alpha[:2], torch.tensor([0, 5], dtype=torch.long), rho),
                lambda: rho_mstep(torch.tensor([0.5, 1.5]), labels[:2], 3, 0.02),
                lambda: rho_mstep(alpha, torch.tensor([0, 1, 2, 3, 9]), 3, 0.02)]:
        _expect((ValueError, TypeError, IndexError, RuntimeError), bad, "bad sparsity input accepted")


# ------------------------------------------------------------------ E-step densities

def test_alpha_1a_bayes_and_bootstrap():
    torch.manual_seed(0)
    lp_y = -3 * torch.rand(16, dtype=torch.float64)          # log p_y in [-3, 0]
    lp_s = -4 * torch.rand(16, dtype=torch.float64)          # log p_s in [-4, 0]
    rho = torch.full((16,), 0.15, dtype=torch.float64)
    a, info = alpha_1a(lp_y, lp_s, rho, T=1.0)
    post = (rho * lp_y.exp()) / (rho * lp_y.exp() + (1 - rho) * lp_s.exp())
    assert torch.allclose(a, post, atol=1e-12), "Eq. 6 != Bayes posterior"
    assert (a > 0).all() and (a < 1).all()
    assert (a >= info["alpha_min"]).all() and (a <= info["alpha_max"]).all(), \
        "Rm 4.2 interior bounds violated"
    assert info["alpha_min"] > 0 and info["alpha_max"] < 1
    # T -> inf: log-ratio/T -> 0  =>  alpha -> sigma(logit rho) = rho (cold-queue bootstrap)
    a_hot, _ = alpha_1a(lp_y, lp_s, rho, T=1e9)
    assert (a_hot - rho).abs().max() < 1e-8, "hot E-step must collapse to the burst prior"
    # grad flows (deterministic function of the model, never detached — Rm 4.1)
    lg = lp_y.clone().requires_grad_(True)
    alpha_1a(lg, lp_s, rho, T=1.0)[0].sum().backward()
    assert lg.grad is not None and torch.isfinite(lg.grad).all()
    for bad in [lambda: alpha_1a(lp_y, lp_s, torch.zeros(16, dtype=torch.float64), 1.0),
                lambda: alpha_1a(lp_y, lp_s, torch.ones(16, dtype=torch.float64), 1.0),
                lambda: alpha_1a(lp_y, lp_s, rho, 0.0),
                lambda: alpha_1a(lp_y[:4], lp_s, rho, 1.0),
                lambda: alpha_1a(torch.empty(0, dtype=torch.float64),
                                 torch.empty(0, dtype=torch.float64),
                                 torch.empty(0, dtype=torch.float64), 1.0)]:
        _expect((ValueError, RuntimeError), bad, "bad alpha_1a input accepted")


def _synthetic_clouds(seed=1):
    torch.manual_seed(seed)
    clouds = {}
    for y, n in ((0, 6), (1, 4), (2, 5), (3, 3)):
        w = torch.rand(n, dtype=torch.float64) + 0.05
        clouds[y] = {"keys": F.normalize(torch.randn(n, D, dtype=torch.float64), dim=1),
                     "weights": w / w.sum()}
    return clouds


def test_cloud_kernel_lse_equals_sinkhorn_cost():
    clouds = _synthetic_clouds()
    torch.manual_seed(2)
    z = F.normalize(torch.randn(10, D, dtype=torch.float64), dim=1)
    eps = 0.5
    lp, costs = cloud_log_p_matrix(z, clouds, eps)
    assert lp.shape == (10, 4) and set(costs) == {0, 1, 2, 3}
    for col, y in enumerate(sorted(clouds)):
        # cost matrix must be EXACTLY 1 - z~^T r_j (Stage 4 reuses this object)
        c_ref = 1.0 - z @ clouds[y]["keys"].t()
        assert torch.allclose(costs[y], c_ref, atol=1e-12), "shared kernel != cost matrix"
        assert costs[y].min() >= -1e-9 and costs[y].max() <= 2.0 + 1e-9
        # logsumexp implementation == direct mixture sum
        direct = (clouds[y]["weights"] * torch.exp(-c_ref / eps)).sum(dim=1).log()
        assert torch.allclose(lp[:, col], direct, rtol=1e-9, atol=1e-12), "LSE != definition of p_y"
    # Rm 4.2 bounds: log p_y <= 0 (weighted avg of exp(-c/eps) <= 1) and >= -2/eps + log(max b)
    assert (lp <= 1e-12).all()
    assert (lp >= -2.0 / eps - 5.0).all(), "log p_y below the B^d uniform bound"
    # gradient side (z_tilde differentiable; clouds are stop-grad material)
    zg = z.clone().requires_grad_(True)
    cloud_log_p_matrix(zg, clouds, eps)[0].sum().backward()
    assert zg.grad is not None and torch.isfinite(zg.grad).all()
    outside = torch.zeros(2, D, dtype=torch.float64)
    outside[0, 0] = 1.5
    bad_clouds = dict(clouds)
    bad_clouds[0] = {"keys": clouds[0]["keys"], "weights": torch.ones(6, dtype=torch.float64)}
    for bad in [lambda: cloud_log_p_matrix(outside, clouds, eps),
                lambda: cloud_log_p_matrix(z, bad_clouds, eps),           # weights not a measure
                lambda: cloud_log_p_matrix(z, {}, eps),                   # empty queue (cold handled upstream)
                lambda: cloud_log_p_matrix(z, clouds, 0.0),
                lambda: cloud_log_p_matrix(z, {0: {"keys": clouds[0]["keys"][:, :3],
                                                   "weights": clouds[0]["weights"]}}, eps)]:
        _expect((ValueError, RuntimeError), bad, "bad cloud input accepted")


def test_log_baseline_gaussian():
    h = torch.tensor([[0.0, 1.0], [2.0, 2.0]], dtype=torch.float64)
    out = log_baseline_gaussian(h, torch.tensor([0.0, 0.0], dtype=torch.float64), sigma_sq=2.0)
    assert torch.allclose(out, torch.tensor([-0.25, -2.0], dtype=torch.float64))
    per = log_baseline_gaussian(h, h, 1.0)
    assert (per == 0).all()
    for bad in [lambda: log_baseline_gaussian(h, torch.zeros(2), 0.0),
                lambda: log_baseline_gaussian(h, torch.zeros(3, 2, dtype=torch.float64), 1.0),
                lambda: log_baseline_gaussian(h.unsqueeze(0), torch.zeros(2), 1.0),
                lambda: log_baseline_gaussian(h + float("nan"), torch.zeros(2), 1.0)]:
        _expect(ValueError, bad, "bad baseline input accepted")


# ------------------------------------------------------------------ 1B marginal + bag baseline

def test_alpha_1b_permutation_invariance():
    torch.manual_seed(3)
    lp = -4 * torch.rand(12, 4, dtype=torch.float64)
    ls = -3 * torch.rand(12, dtype=torch.float64)
    rho = torch.tensor([0.3, 0.05, 0.2, 0.4], dtype=torch.float64)
    a = alpha_1b(lp, ls, rho)
    assert (a > 0).all() and (a < 1).all()
    perm = torch.tensor([2, 0, 3, 1])
    a_p = alpha_1b(lp[:, perm], ls, rho[perm])
    assert torch.allclose(a, a_p, atol=1e-12), "Eq. 14 must be label-free (class-permutation invariant)"
    # closed-form Bayes with uniform class prior (Prop 10.1)
    rho_bar = rho.mean()
    num = lp.exp() @ rho / 4
    post = num / (num + (1 - rho_bar) * ls.exp())
    assert torch.allclose(a, post, atol=1e-12), "Eq. 14 != Bayes marginal"
    for bad in [lambda: alpha_1b(lp, ls, rho[:3]),
                lambda: alpha_1b(lp, ls, torch.ones(4, dtype=torch.float64)),
                lambda: alpha_1b(lp[0], ls, rho),
                lambda: alpha_1b(torch.empty(0, 4, dtype=torch.float64),
                                 torch.empty(0, dtype=torch.float64), rho)]:
        _expect((ValueError, RuntimeError), bad, "bad alpha_1b input accepted")


def _synthetic_bag(seed=7, tb=36, nb=4):
    """Baseline bulk near [0.5,0,..] with a few far burst rows."""
    torch.manual_seed(seed)
    base = torch.zeros(tb - nb, 4, dtype=torch.float64)
    base[:, 0] = 0.5
    base += 0.02 * torch.randn(tb - nb, 4, dtype=torch.float64)
    burst = torch.zeros(nb, 4, dtype=torch.float64)
    burst[:, 0] = burst[:, 1] = 1.9
    h = torch.cat([base, burst])
    lp = torch.full((tb, 2), -8.0, dtype=torch.float64)
    lp[tb - nb:] = 5.0
    return h, lp


def test_two_step_baseline_certificate():
    h, lp = _synthetic_bag()
    rho = torch.tensor([0.1, 0.2], dtype=torch.float64)
    res = two_step_baseline(h, lp, rho, sigma_sq=1.0)
    assert 0.0 < res["kappa"] < 1.0, "contraction certificate must be a number in (0,1)"
    assert res["gap"] <= res["gap_theory"] + 1e-9, "bound (16) first inequality violated"
    assert res["gap"] <= res["bound"] + 1e-9, "bound (16) second inequality violated"
    assert res["bound"] >= res["gap_theory"] - 1e-12
    base_mean = h[:32].mean(dim=0)
    assert (res["mu_hat"] - base_mean).norm() < 0.1, "G^2 baseline drifted off the bulk"
    a = res["alpha"]
    assert (a > 0).all() and (a < 1).all()
    assert a[:32].mean() < 0.05 and a[32:].mean() > 0.8, "alpha^(1B) must separate bulk vs bursts"
    # alpha returned must be Eq. 14 evaluated AT mu_hat (R_b feeds on that pair)
    lp_s_hat = log_baseline_gaussian(h, res["mu_hat"], 1.0)
    re_eval = alpha_1b(lp, lp_s_hat, rho)
    assert torch.allclose(a, re_eval, atol=1e-12), "alpha not evaluated at mu_hat_b = G^2(h_bar')"
    # R_b (Eq. 17) manual
    r_b = bag_statistic(a, h)
    manual = (a.unsqueeze(1) * h).sum(0) / h.shape[0]
    assert torch.allclose(r_b, manual, atol=1e-12)
    # batch wrapper + mixed T_b
    h2, lp2 = _synthetic_bag(seed=8, tb=20, nb=2)
    outs = batch_two_step_baselines([h, h2], [lp, lp2], rho, sigma_sq=1.0)
    assert len(outs) == 2 and all(o["kappa"] < 1.0 for o in outs)
    # fail-loud: fully burst-flagged bag has no baseline mass
    lp_all = torch.full((36, 2), 60.0, dtype=torch.float64)
    _expect(ValueError, lambda: two_step_baseline(h, lp_all, rho, sigma_sq=1.0),
            "fully-burst bag accepted (sum(1-alpha) ~ 0)")
    for bad in [lambda: two_step_baseline(h[0], lp, rho, 1.0),                 # 1-D bag
                lambda: two_step_baseline(h, lp[:10], rho, 1.0),               # T_b mismatch
                lambda: two_step_baseline(h, lp, rho[:1], 1.0),
                lambda: two_step_baseline(h, lp, rho, 0.0),
                lambda: two_step_baseline(h, lp, torch.zeros(2, dtype=torch.float64), 1.0),
                lambda: two_step_baseline(h, lp, rho, 1.0, max_iter=0),
                lambda: batch_two_step_baselines([h], [lp, lp2], rho, 1.0),
                lambda: batch_two_step_baselines([], [], rho, 1.0)]:
        _expect(ValueError, bad, "bad two_step input accepted")


def test_bag_statistic_validation():
    a = torch.tensor([0.5, 1.0], dtype=torch.float64)
    h = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    r = bag_statistic(a, torch.tensor([0.5, 1.0], dtype=torch.float64).unsqueeze(1).expand(2, 4))
    assert torch.allclose(r, torch.tensor([0.625, 0.625, 0.625, 0.625], dtype=torch.float64))
    _expect(ValueError, lambda: bag_statistic(a[:1], h), "T_b mismatch accepted")
    _expect(ValueError, lambda: bag_statistic(torch.tensor([0.5, 1.4]),
                                              h[:2].contiguous()), "alpha>1 accepted")
    _expect(ValueError, lambda: bag_statistic(torch.empty(0, dtype=torch.float64),
                                              torch.empty(0, 4, dtype=torch.float64)), "T_b=0 accepted")


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
