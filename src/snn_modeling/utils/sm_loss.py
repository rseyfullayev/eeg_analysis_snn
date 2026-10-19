"""SMC loss engine — Eq. 23 assembled objective (fixed.pdf §11, plan §5 Stage 6).

    L_total = L_anc(z~) + lambda_OT * L_em + lambda_sub * L_sub + lambda_rho * L_spr

orchestrating the tested Stage 1-5 primitives: NullspaceState (shared B^d
geometry + subject M-step), SupMoCoQueue (dual readouts), Option-B priors +
E-step alpha^(1A) + rho self-consistency (Eq. 6/12), projected DCL anchor +
certificate C_K (Eq. 22, Thm 7.3), readiness gate + EMA lambda_OT (Eq. 13,
§8.3), debiased dustbin-Sinkhorn emotion term (Eq. 19/Def. 5.4) and the
sparsity prior (Eq. 21). Nothing is detached silently: the gradient routing
is explicit (queue/cloud/coupling/rho/U/ema tables are stop-grad M-step
material; gradients reach the encoder through z~ in L_anc, L_em, L_sub and
through alpha in L_spr).

Window-id plumbing (§9.4): the dataset gives per-trial local window ids
(timestamps, with -1 pads). The engine composes globally-unique queue UIDs

    uid = trial_id * UID_SHIFT + local_window_id        (UID_SHIFT = 2**20)

so same-window masking (queue reads) and LOTO ghosting (trial = uid >> 20)
both work off the single window_ids field the Stage 2 queue already stores.

Bootstrap order (§11.4ii) is asserted, not assumed: before the queue holds
anything the anchor/emotion terms are defined-zero (cold flags), alpha
falls back to the seeded rho (T is at its annealing maximum), the gate
sits at g_floor and lambda_OT = 0. Every §12 inequality runs loudly:
Option-B contraction (inside option_b_seed), eps admissibility at seed
time, c-gate at config time, queue sizing + staleness via run_sizing_checks.
"""

import math

import torch
import torch.nn as nn

from .space import NullspaceState
from ..memory.queue import (SupMoCoQueue, check_sizing, check_staleness, queue_ceiling,
                            queue_floor)
from .sparsity import (burst_delta_scores, option_b_seed, rho_mstep, sparsity_loss)
from .estep import (alpha_1a, cloud_log_p_matrix, log_baseline_gaussian)
from .sinkhorn import (burst_marginal, class_ot_term, dustbin_price, check_eps_admissible,
                       eps_lower_bound)
from .dcl import ProjectedAnchor
from .gates import OTCoefficient, check_c_gate, grad_norm, readiness_gate

__all__ = ["SMCLossEngine", "LOCKPASS_SCHEMA"]

LOCKPASS_SCHEMA = "smc-lockpass-v1"
UID_SHIFT = 2 ** 20


def _get(cfg, dotted, default):
    """Dotted-path lookup over nested dict/OmegaConf; returns `default` when the
    path is absent (required knobs simply pass default=None and callers validate)."""
    node = cfg
    for part in dotted.split("."):
        node = node.get(part, None) if hasattr(node, "get") else None
        if node is None:
            return default
    return node


class SMCLossEngine(nn.Module):
    """Stateful Eq. 23 orchestrator. Children (NullspaceState, SupMoCoQueue) and
    the rho/step buffers ride in state_dict(); the OTCoefficient float state goes
    through sm_state()/load_sm_state() alongside it."""

    def __init__(self, cfg, feature_dim, num_classes, num_subjects, total_steps):
        super().__init__()
        if feature_dim < 2 or int(num_classes) < 2 or int(num_subjects) < 2:
            raise ValueError(f"SMCLossEngine: dims invalid ({feature_dim}/{num_classes}/{num_subjects})")
        if int(total_steps) < 1:
            raise ValueError(f"SMCLossEngine: total_steps must be >= 1, got {total_steps}")
        g = lambda k, d=None: _get(cfg, k, d)

        # ---- knobs (approved standard defaults; all config-tunable) ----------
        self.total_steps = int(total_steps)
        self.num_classes = int(num_classes)
        self.tau_norm = float(g("tau_norm", 0.25))
        self.rank_band = float(g("rank_energy_band", 0.85))
        self.warmup_epochs = int(g("warmup_epochs", 2))
        self.space_momentum = float(g("space_momentum", 0.9))
        self.queue_momentum = float(g("queue_momentum", 0.9))  # (staleness knob, §9.4)
        self.rho_min = float(g("rho_min", 0.02))
        # §13 row-2 Failure-Catalogue switch: pinned huge drop price => the dustbin
        # column is inert => plain balanced OT (SCDA subject-mining pathology).
        # Intentionally OUT-OF-MODEL (Thm 12.2's tau floor is superseded) —
        # evidence-only and announced at build, never a silent mode.
        self.force_balanced = bool(g("force_balanced", False))
        if self.force_balanced:
            print("[SMCLossEngine] ABLATION ON: tau_y pinned to 1e6 (balanced-OT pathology, "
                  "§13 row 2) — results from this run are paper evidence, NOT the model.")
        self.seed_s = float(g("seed.s", 2.0))
        self.seed_log_coeff = float(_get(cfg, "seed.log_coeff", float(feature_dim)))
        self.sigma_sq = float(g("sigma_sq", 1.0))
        self.tau_dcl = float(g("tau_dcl", 0.1))
        self.anchor_on = str(g("anchor_on", "projected"))
        if self.anchor_on not in ("projected", "raw"):
            raise ValueError(f"SMCLossEngine: anchor_on must be 'projected'/'raw' (§11.4i), got {self.anchor_on!r}")
        self.t_anneal = float(g("anneal.t_start", 20.0))
        self.t_end = float(g("anneal.t_end", 1.0))
        self.anneal_frac = float(g("anneal.frac", 0.3))
        if not (self.t_anneal > self.t_end > 0.0) or not (0.0 < self.anneal_frac <= 1.0):
            raise ValueError(f"SMCLossEngine: bad anneal schedule {self.t_anneal}->{self.t_end} "
                             f"over {self.anneal_frac} of steps")
        self.c = check_c_gate(g("c", 0.5), c_min=g("c_min", 0.05), c_max=g("c_max", 0.9))
        self.g_floor = float(g("g_floor", 0.1))
        if not (0.0 <= self.g_floor < 1.0):
            raise ValueError(f"SMCLossEngine: g_floor must lie in [0,1), got {self.g_floor}")
        self.lambda_anc = float(g("lambda_anc", 1.0))
        self.lambda_sub = float(g("lambda_sub", 1.0))
        self.lambda_rho = float(g("lambda_rho", 0.1))
        if self.lambda_anc < 0 or self.lambda_sub < 0 or self.lambda_rho < 0:
            raise ValueError("SMCLossEngine: lambda_anc/lambda_sub/lambda_rho must be >= 0")
        self.sn_tol = float(g("sinkhorn.tol", 1e-3))
        self.sn_T_sc = int(g("sinkhorn.T_sc", 30))
        self.sn_iters = int(g("sinkhorn.n_iters", 5000))
        self.sn_solver_tol = float(g("sinkhorn.solver_tol", 1e-10))
        eps_max = float(g("eps_max", 1.0))  # Thm 12.2 upper end (blur budget); with the
        #  approved tol=1e-3/T_sc=30 defaults the lower bound is ~0.92, so a midpoint
        #  default of 0.5 would make the admissible interval EMPTY (caught at build).
        eps = _get(cfg, "eps", None)
        self.eps = None if eps is None else float(eps)
        if self.eps is not None and (self.eps <= 0 or self.eps > eps_max):
            raise ValueError(f"SMCLossEngine: eps={self.eps} outside (0, eps_max={eps_max}] "
                             "(Thm 12.2 interval upper end)")
        self._eps_max = eps_max

        # ---- components ------------------------------------------------------
        self.space = NullspaceState(feature_dim=feature_dim, num_subjects=int(num_subjects),
                                    tau_norm=self.tau_norm, ema_momentum=self.space_momentum,
                                    rank_energy_band=self.rank_band,
                                    warmup_epochs=self.warmup_epochs)
        self.queue = SupMoCoQueue(int(g("n_q", 4096)), feature_dim, num_classes=self.num_classes)
        self.anchor = ProjectedAnchor(tau_dcl=self.tau_dcl, anchor_on=self.anchor_on)
        self.coeff = OTCoefficient(momentum=float(g("gate_ema_momentum", 0.9)))

        self.register_buffer("rho", torch.full((self.num_classes,), float("nan")))
        self.register_buffer("seeded", torch.zeros(1, dtype=torch.bool))
        self.register_buffer("step", torch.zeros(1, dtype=torch.long))
        self._nu_cache = {}
        self._nu_cache_key = None

    # ------------------------------------------------------------------ priors

    def seed_priors(self, norm_sq_by_class):
        """§9.5 Option-B: one threshold-free fixed-point seed per class from the
        raw nullspace-embedding squared norms (pre-pass over training data).
        Also resolves/validates eps (Thm 12.2 lower bound uses the seeded tau_y).
        Returns a per-class diagnostic dict."""
        if len(norm_sq_by_class) != self.num_classes:
            raise ValueError(f"seed_priors: need all {self.num_classes} classes, got "
                             f"{sorted(norm_sq_by_class) if isinstance(norm_sq_by_class, dict) else 'list'}")
        rho = torch.empty(self.num_classes, dtype=torch.float64)
        diags = {}
        for y in range(self.num_classes):
            norm_sq = torch.as_tensor(norm_sq_by_class[y], dtype=torch.float64)
            delta = burst_delta_scores(norm_sq.float(), s=self.seed_s, log_coeff=self.seed_log_coeff)
            r, diag = option_b_seed(delta, self.rho_min)
            rho[y] = r
            diags[y] = diag
        self.rho.copy_(rho.to(self.rho.dtype))
        if self.eps is None:  # approved-default midpoint of the Thm 12.2 interval
            lb = max(eps_lower_bound(self.sn_tol, self.sn_T_sc,
                                     float(dustbin_price(0.5, float(rho[y])))) for y in range(self.num_classes))
            # NOTE: tau_y = eps*h(rho_y) scales with eps; solve lb(eps)=eps self-consistently
            # by scanning the midpoint definition eps = (lb(eps)+eps_max)/2 with 3 fixed-point steps.
            eps = 0.5 * (lb + self._eps_max)
            for _ in range(3):
                lb_e = max(eps_lower_bound(self.sn_tol, self.sn_T_sc,
                                           float(dustbin_price(eps, float(rho[y]))))
                           for y in range(self.num_classes))
                eps = 0.5 * (lb_e + self._eps_max)
            self.eps = float(eps)
        for y in range(self.num_classes):  # hard gate per class, per §12.2
            check_eps_admissible(self.eps, self.sn_tol, self.sn_T_sc,
                                 float(dustbin_price(self.eps, float(rho[y]))))
        self.seeded.fill_(True)
        return {"rho": {y: float(rho[y]) for y in range(self.num_classes)},
                "diags": diags, "eps": self.eps}

    # ------------------------------------------------------------------ init asserts

    def run_sizing_checks(self, batch_size, epochs, steps_per_epoch, L_const=1.0,
                          d_eff=None, tol_eff=0.1, C_const=4.0, eps_tol=1e-2):
        """§9.6 floor/ceiling + §9.4 staleness bound for the ACTUAL run budget —
        called by train.py once shapes/steps are known. Loud, no auto-shrink."""
        floor = queue_floor(C_const=C_const, d_eff=float(d_eff or self.space.feature_dim),
                            tol=tol_eff, rho_tilde_min=self.rho_min)
        ceiling = queue_ceiling(eps_tol=eps_tol, B=batch_size,
                                T_total=int(epochs * steps_per_epoch),
                                L=L_const, D=float(self.space.feature_dim))
        check_sizing(self.queue.queue_size, floor, ceiling)
        check_staleness(self.queue.queue_size, batch_size, self.queue_momentum,
                        L_const, float(self.space.feature_dim), eps_tol)
        return floor, ceiling

    # ------------------------------------------------------------------ helpers

    def _subject_baseline(self, subjects):
        """mu_hat'_s = (I-UU^T)(mu_bar_s - mu): detached M-step tables (Rm 6.5)."""
        return self.space.apply_projector(self.space.ema_mean[subjects] - self.space.mu).detach()

    @staticmethod
    def _compose_uids(trial_ids, window_ids):
        if trial_ids.dtype != torch.long or window_ids.dtype != torch.long:
            raise ValueError("uids require long trial_ids/window_ids")
        if (trial_ids < 0).any():
            raise ValueError(f"trial_ids must be >= 0 (got min {int(trial_ids.min())})")
        if (window_ids < 0).any():
            raise ValueError(
                f"local window_ids must be >= 0 — the -1 pad sentinel must be filtered "
                f"by the caller before the SMC engine (found {int((window_ids < 0).sum())} pads)")
        if (window_ids >= UID_SHIFT).any():
            raise ValueError("local window_ids exceed UID_SHIFT — choose a larger shift, "
                             "do NOT silently mod-fold collisions")
        uids = trial_ids * UID_SHIFT + window_ids
        if torch.unique(uids).numel() != uids.numel():
            raise ValueError("duplicate (trial, window) uids within one batch — dataset bug")
        return uids

    def _resolve_eps(self):
        if not bool(self.seeded.item()):
            raise RuntimeError("SMCLossEngine.forward before seed_priors() — rho would be NaN "
                               "and the burst prior seeds the E-step (no silent cold uniform)")
        if self.eps is None:
            raise RuntimeError("eps unresolved — seed_priors() must run first (it validates the "
                               "Thm 12.2 interval)")
        return self.eps

    # ------------------------------------------------------------------ forward

    def forward(self, z, key_z, labels, subjects, window_ids, trial_ids):
        """One window-level SMC step (Eq. 23).

        z, key_z:  (B,D) UNIT-NORM online / momentum embeddings (key_z may carry
                   no grad — it is stop-grad material; engine re-projects both).
        labels, subjects: (B,) long in [0,C)/[0,S).
        window_ids: (B,) long LOCAL window ids (>= 0, pads pre-filtered).
        trial_ids:  (B,) long bag/trial ids (>= 0).
        Returns dict with 'total' (grad tensor) + parts + diagnostics.
        """
        eps = self._resolve_eps()
        B = z.shape[0]
        if B == 0:
            raise ValueError("SMCLossEngine: empty batch (B14 discipline)")
        for name, t in (("labels", labels), ("subjects", subjects),
                        ("window_ids", window_ids), ("trial_ids", trial_ids)):
            if t.shape != (B,):
                raise ValueError(f"SMCLossEngine: {name} must be (B,)={B}, got {tuple(t.shape)}")
        if labels.dtype != torch.long or subjects.dtype != torch.long:
            raise ValueError("SMCLossEngine: labels/subjects must be long")
        if (labels < 0).any() or labels.max() >= self.num_classes:
            raise ValueError("SMCLossEngine: labels outside [0,C)")
        if (subjects < 0).any() or subjects.max() >= self.space.num_subjects:
            raise ValueError("SMCLossEngine: subjects outside [0,S)")
        uids = self._compose_uids(trial_ids, window_ids)

        step = int(self.step.item())
        T = max(self.t_end, self.t_anneal - (self.t_anneal - self.t_end)
                * min(1.0, step / max(1, self.anneal_frac * self.total_steps)))

        # shared geometry (fixed.pdf §2): single project() path for every consumer
        h_q, zt_q, _ = self.space.project(z)                 # grad-carrying side
        with torch.no_grad():
            _, zt_k, _ = self.space.project(key_z.detach())  # stop-grad key path

        clouds = self.queue.read_class_clouds(exclude_window_ids=uids)
        neg_keys, _, _, neg_uids = self.queue.read_negatives(exclude_window_ids=uids)

        # ---- anchor (Eq. 22) + bootstrap cold path (§11.4ii: defined zero, flagged)
        cold_anchor = neg_keys.shape[0] == 0
        if cold_anchor:
            L_anc = zt_q.sum() * 0.0
            C_K, K = 0.0, 1
        else:
            anc = self.anchor(zt_q, zt_k, neg_keys,
                              query_group=trial_ids, neg_group=neg_uids // UID_SHIFT)
            L_anc, C_K, K = anc["L_DCL"], float(anc["C_K"].detach()), int(anc["K"])

        # ---- E-step responsibilities alpha^(1A) (Eq. 6) with cold-class fallback
        rho_vec = self.rho.detach()
        alpha = rho_vec[labels].clone()
        cold_rows = B
        ainfo = None
        if clouds:
            classes = sorted(int(y) for y in clouds)
            col_of = {y: i for i, y in enumerate(classes)}
            present = torch.as_tensor([col_of.get(int(l), -1) for l in labels.tolist()],
                                      dtype=torch.long, device=z.device)
            rows = (present >= 0).nonzero(as_tuple=True)[0]
            cold_rows = B - int(rows.numel())
            if rows.numel() > 0:
                lp, _costs = cloud_log_p_matrix(zt_q, clouds, eps)
                log_p_y = lp[rows, present[rows]]
                log_p_s = log_baseline_gaussian(h_q, self._subject_baseline(subjects), self.sigma_sq)
                a_present, ainfo = alpha_1a(log_p_y, log_p_s[rows], rho_vec[labels[rows]], T)
                alpha = alpha.clone()
                alpha[rows] = a_present

        # ---- sparsity prior (Eq. 21) with the CURRENT rho table, then Eq. 12 M-step
        L_spr = sparsity_loss(alpha, labels, rho_vec)
        with torch.no_grad():
            self.rho.copy_(rho_mstep(alpha.detach(), labels, self.num_classes,
                                     self.rho_min, prev_rho=self.rho))

        # ---- subject residual (Eq. 20), U/tables detached (plan §5 Stage 6);
        #      U is FORCED ZERO during warmup epochs (§11.4ii "U = 0 held fixed")
        r = int(self.space.rank.item()) if not self.space.warmup_active else 0
        U = self.space.U[:, :r].detach() if r > 0 else self.space.U[:, :0]
        resid = (z - self.space.mu.detach())
        if U.shape[1] > 0:
            a_s = (self.space.ema_mean[subjects].detach() - self.space.mu.detach()) @ U
            resid = resid - a_s @ U.t()
        L_sub = ((1.0 - alpha.detach()) * (resid ** 2).sum(dim=1)).mean()

        # ---- emotion term (Eq. 19): per-class debiased S_eps, nu-self cached
        cold_em = True
        L_em = zt_q.sum() * 0.0
        d_norm, ess_used = {}, {}
        if clouds:
            cache_key = (int(self.queue.ptr.item()), int(self.queue.filled.item()))
            if self._nu_cache_key != cache_key:
                self._nu_cache = {}
                self._nu_cache_key = cache_key
            for y in sorted(int(l) for l in labels.unique().tolist()):
                if y not in clouds:
                    continue  # class not yet in the queue: emotion term inert for it (logged)
                rows = (labels == y).nonzero(as_tuple=True)[0]
                if rows.numel() == 0:
                    continue
                a_y = burst_marginal(alpha.detach()[rows])
                cl = clouds[y]
                tau_y = 1.0e6 if self.force_balanced else float(
                    dustbin_price(eps, float(self.rho[y])))
                out = class_ot_term(zt_q[rows], a_y, cl["keys"], cl["weights"],
                                    eps, tau_y, self_nu_cache=self._nu_cache.get(y),
                                    n_iters=self.sn_iters, tol=self.sn_solver_tol)
                L_em = L_em + out["S"]
                self._nu_cache[y] = out["self_nu_cache"]
                d_norm[y] = out["diagnostics"]["D_norm_y"]
                cold_em = False

        # ---- gate + lambda_OT (Eq. 13, §8.3): both norms w.r.t. the SAME z~
        # cold anchor has no K>=2 to normalize by -> gate sits at floor (§11.4ii)
        g = self.g_floor if cold_anchor else readiness_gate(C_K, K, self.g_floor)
        anc_n = grad_norm(L_anc, zt_q)
        em_n = grad_norm(L_em, zt_q)
        self.coeff.observe(anc_n, em_n)
        lam = self.coeff.value(self.c, g)

        total = self.lambda_anc * L_anc + lam * L_em + self.lambda_sub * L_sub + self.lambda_rho * L_spr

        # ---- Table 1 diagnostics, computed BEFORE the M-step side effects so the
        #      (µ, U) used in the denominator is the EXACT geometry that produced
        #      h_q: rank 0 ⇒ energy_ratio = 1.0 by construction (#3 semantics).
        #      #4 intra-trial Stdt(α_t): per-trial std over windows, averaged;
        #      <0.02 once the anneal is at its floor means α ≡ ρ (escape).
        with torch.no_grad():
            t_dev = trial_ids.to(alpha.device)
            a_flat = alpha.detach()
            _stds = [a_flat[t_dev == t].std(unbiased=False)
                     for t in torch.unique(t_dev) if int((t_dev == t).sum()) >= 2]
            stdt_intra = float(torch.stack(_stds).mean()) if _stds else 0.0
            e_full = (z.detach() - self.space.mu.detach()).pow(2).sum(dim=1).mean()
            e_null = h_q.detach().pow(2).sum(dim=1).mean()
            energy_ratio = float(e_null / e_full) if float(e_full) > 1e-12 else 0.0

        # ---- M-step side effects (stop-grad material, Prop 8.4 / §6)
        with torch.no_grad():
            self.space.update_batch(z.detach(), alpha.detach(), subjects)
            self.queue.enqueue(zt_k, labels, alpha.detach(), uids, step)
            self.step += 1

        diag = {"step": step, "T": T, "gate": g, "lambda_ot": lam,
                "stdt_intra": stdt_intra, "energy_ratio": energy_ratio,
                "anneal_complete": bool(step >= self.anneal_frac * self.total_steps),
                "C_K": C_K, "K": K, "n_negatives": int(neg_keys.shape[0]),
                "cold_anchor": cold_anchor, "cold_em": cold_em,
                "cold_alpha_rows": cold_rows,
                "alpha_mean": float(alpha.detach().mean()),
                "alpha_std": float(alpha.detach().std()) if B > 1 else 0.0,
                "D_norm": d_norm, "queue": self.queue.summary(),
                "space": self.space.summary(), "eps": self.eps,
                "ess": {int(k_): v for k_, v in self.queue.ess_by_class().items()}}
        return {"total": total, "L_anc": L_anc, "L_em": L_em, "L_sub": L_sub,
                "L_spr": L_spr, "alpha": alpha, "diagnostics": diag}

    # ------------------------------------------------------------------ lifecycle

    def on_epoch_end(self):
        self.space.advance_epoch()

    def sm_state(self):
        return {"ot_coeff": self.coeff.state_dict(), "step": int(self.step.item()),
                "eps": self.eps}

    def load_sm_state(self, payload):
        self.coeff.load_state_dict(payload["ot_coeff"])
        self.step.fill_(int(payload["step"]))
        if payload.get("eps") is not None:
            self.eps = float(payload["eps"])
        return self

    def lock_fullpass(self):
        """End-of-Phase-1A lock artifact (§6.4/§9.3): exact (mu, U, rank), rho,
        final beta-weighted clouds — everything Phase 1B is evaluated against."""
        if not bool(self.seeded.item()):
            raise RuntimeError("lock_fullpass before seed_priors — refusing an artifact with NaN rho")
        clouds = self.queue.read_class_clouds()
        payload = {
            "schema": LOCKPASS_SCHEMA,
            "space": self.space.lock_payload(),
            "rho": self.rho.detach().cpu().clone(),
            "clouds": {y: {"keys": c["keys"].detach().cpu().clone(),
                           "weights": c["weights"].detach().cpu().clone()}
                       for y, c in clouds.items()},
            "step": int(self.step.item()),
            "eps": self.eps,
        }
        validate_lockpass(payload)
        return payload


def validate_lockpass(payload):
    if not isinstance(payload, dict) or payload.get("schema") != LOCKPASS_SCHEMA:
        raise ValueError(f"lockpass schema must be {LOCKPASS_SCHEMA!r}")
    rho = payload["rho"]
    if rho.dim() != 1 or rho.numel() < 2 or not torch.isfinite(rho).all():
        raise ValueError("lockpass rho must be a finite vector")
    if (rho <= 0).any() or (rho >= 1).any():
        raise ValueError("lockpass rho outside (0,1) — interiority violated")
    if not payload["clouds"]:
        raise ValueError("lockpass has empty clouds — 1A never filled the queue?")
    for y, c in payload["clouds"].items():
        for t, name in ((c["keys"], "keys"), (c["weights"], "weights")):
            if not torch.isfinite(t).all():
                raise ValueError(f"lockpass cloud {y} {name} non-finite")
        if abs(float(c["weights"].sum()) - 1.0) > 1e-4:
            raise ValueError(f"lockpass cloud {y}: weights are not a probability measure")
    return payload
