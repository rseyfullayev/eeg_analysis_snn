"""Phase 1B R_b readout adapter — fixed.pdf §10.2 (Eq. 17/18, Props 10.3/10.4),
§11.2/§11.4(iii); plan §5 Stage 7.

Consumes a `smc-lockpass-v1` artifact (frozen mu/U/rank, rho, beta-weighted
class clouds, eps) plus per-bag UNIT-NORM window embeddings z (T_b, D) — the
v2 bag payload's stored tensor. h'/z~ come ONLY through NullspaceState.project
(the single geometry path; Rm 6.5 makes unseen subjects work by construction —
the `subject` field of a payload is deliberately never read here: Phase 1B is
label- AND subject-free at inference).

Per bag (Prop 10.1/10.2, Eq. 14-16):
    log p_y(z~) from the frozen clouds; baseline mu_hat_b = G^2(h_bar') via the
    CERTIFIED two_step_baseline (kappa >= 1 raises, never a silent extra step);
    R_b = (1/T_b) sum_t alpha_t^(1B) h'_t  (Eq. 17, NOT normalized by sum alpha
    — magnitude keeps the burst rate; ||R_b|| <= 2 by construction).

Classifier (Prop 10.4 + Table 1 #2), in the mandated order:
    1. zero-train nearest centroid  argmin_y ||R_b - R_bar_y||^2,
    2. ONE nn.Linear(D, C, bias=True) Bayes warm-start (Eq. 18):
           w_y = rho_y mu_y / sigma_R^2  =  R_bar_y / sigma_R^2   (since E[R|y] = rho_y mu_y)
           b_y = -||rho_y mu_y||^2/(2 sigma_R^2) + log(1/C)
       then CE with L2 weight decay lambda_wd in [1e-2, 1e-1].
    No cosine head (divides the burst rate out), no contrastive/prototype/
    attention branches, no temperature ensemble (Rm 10.5).

Rolling-buffer baseline (§11.4(iii)) is available behind mode="rolling_buffer"
with an explicit same-subject session buffer; default is per-bag G^2.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.space import NullspaceState
from ..utils.estep import (alpha_1b, bag_statistic, cloud_log_p_matrix,
                           log_baseline_gaussian, two_step_baseline)
from ..utils.sm_loss import validate_lockpass
from ..utils.features import load_bag_payload

__all__ = ["Phase1BReadout", "run_phase_1b", "bags_from_payloads", "HEAD_SCHEMA"]

HEAD_SCHEMA = "smc-mil-head-v1"


class _Cloud(nn.Module):
    """One frozen class cloud as buffers so .to(device)/state_dict just work."""

    def __init__(self, keys, weights):
        super().__init__()
        self.register_buffer("keys", keys.float().clone())
        self.register_buffer("weights", weights.float().clone())


def bags_from_payloads(paths):
    """Strict v2 payload reader -> (z_list, labels, bag_ids). Subjects are read
    for reporting only; the R_b pipeline never consumes them (subject-free by
    construction, Rm 6.5)."""
    z_list, labels, bag_ids = [], [], []
    for p in paths:
        obj = load_bag_payload(p)
        z_list.append(obj["z"])
        labels.append(int(obj["label"]))
        bag_ids.append(int(obj["bag_id"]))
    return z_list, torch.tensor(labels, dtype=torch.long), bag_ids


class Phase1BReadout(nn.Module):
    """Frozen geometry + clouds + certified bag statistics + Eq. 18 head."""

    def __init__(self, payload, sigma_sq=1.0, two_step_max_iter=64, two_step_tol=1e-12,
                 omit_duration_norm=False):
        super().__init__()
        # Failure-Catalogue ablation switch (§13 row "Omit 1/T_b"): default False IS
        # the spec behaviour (Eq. 17); enabling it produces paper evidence, loudly.
        if not isinstance(omit_duration_norm, bool):
            raise TypeError(f"omit_duration_norm must be bool, got {omit_duration_norm!r}")
        self.omit_duration_norm = omit_duration_norm
        if omit_duration_norm:
            print("[Phase1BReadout] ABLATION ON: R_b = sum alpha_t h'_t WITHOUT 1/T_b - "
                  "trial-duration bias is the EXPECTED outcome (§13 row 8), not a bug.")
        validate_lockpass(payload)
        rho = payload["rho"]
        C = int(rho.numel())
        clouds = payload["clouds"]
        missing = [y for y in range(C) if y not in clouds]
        if missing:
            raise ValueError(f"lockpass clouds missing classes {missing} — Phase 1A never "
                             "put them in the queue; Phase 1B cannot score them (no silent skip)")
        extra = [y for y in clouds if not (0 <= int(y) < C)]
        if extra:
            raise ValueError(f"lockpass clouds carry out-of-range classes {extra} — corrupt artifact")
        eps = payload.get("eps")
        if eps is None or not (float(eps) > 0) or not math.isfinite(float(eps)):
            raise ValueError(f"lockpass eps must be positive/finite, got {eps!r}")
        if not (sigma_sq > 0) or not math.isfinite(sigma_sq):
            raise ValueError(f"sigma_sq must be positive/finite, got {sigma_sq}")

        sp = payload["space"]
        self.space = NullspaceState(feature_dim=sp["feature_dim"], num_subjects=sp["num_subjects"],
                                    tau_norm=sp["tau_norm"], ema_momentum=sp["ema_momentum"],
                                    rank_energy_band=sp["rank_energy_band"],
                                    warmup_epochs=sp["warmup_epochs"])
        self.space.load_lock(sp)   # strict: schema, dims, orthonormality
        self.eps = float(eps)
        self.sigma_sq = float(sigma_sq)
        self.ts_max_iter = int(two_step_max_iter)
        self.ts_tol = float(two_step_tol)
        self.num_classes = C
        self.register_buffer("rho", rho.float().clone())
        self.clouds = nn.ModuleList([_Cloud(clouds[y]["keys"], clouds[y]["weights"])
                                     for y in range(C)])
        self.register_buffer("centroids", torch.zeros(C, sp["feature_dim"]))
        self.register_buffer("sigma_R_sq", torch.zeros(()))
        self.head = None
        self._fitted = False

    # ------------------------------------------------------------------ geometry

    @torch.no_grad()
    def _bag_geometry(self, z):
        if z.dim() != 2 or z.shape[1] != self.space.feature_dim:
            raise ValueError(f"bag z must be (T_b, {self.space.feature_dim}), got {tuple(z.shape)}")
        z = z.float().to(self.rho.device)
        h, zt, _hn = self.space.project(z)            # validates unit-norm rows
        lp, _costs = cloud_log_p_matrix(zt, {y: {"keys": c.keys, "weights": c.weights}
                                             for y, c in enumerate(self.clouds)}, self.eps)
        return h, lp

    @staticmethod
    def _check_labels(labels, n, C):
        if labels is None:
            return None
        labels = torch.as_tensor(labels, dtype=torch.long).view(-1)
        if labels.numel() != n:
            raise ValueError(f"labels length {labels.numel()} != {n} bags")
        if n and ((labels < 0).any() or labels.max() >= C):
            raise ValueError(f"labels outside [0,{C})")
        return labels

    @torch.no_grad()
    def encode_bag(self, z, mode="per_bag", buffer_z=None):
        """One bag -> R_b + certification diagnostics (kappa/bound/gap, Eq. 16)."""
        if mode not in ("per_bag", "rolling_buffer"):
            raise ValueError(f"encode_bag: mode must be per_bag|rolling_buffer, got {mode!r}")
        if mode == "per_bag" and buffer_z is not None:
            raise ValueError("encode_bag: buffer_z only valid for rolling_buffer mode")
        if mode == "rolling_buffer" and (buffer_z is None or len(buffer_z) == 0):
            raise ValueError("encode_bag: rolling_buffer mode needs a non-empty same-subject "
                             "session buffer (§11.4iii), got none")
        h, lp = self._bag_geometry(z)
        if mode == "per_bag":
            tb = two_step_baseline(h, lp, self.rho, self.sigma_sq,
                                   max_iter=self.ts_max_iter, tol=self.ts_tol)
            alpha, kappa, bound, gap = tb["alpha"], tb["kappa"], tb["bound"], tb["gap"]
        else:
            hb = [h]; lb = [lp]
            for zb in buffer_z:
                hz, lz = self._bag_geometry(zb)
                hb.append(hz); lb.append(lz)
            h_ext = torch.cat(hb, dim=0)
            lp_ext = torch.cat(lb, dim=0)
            tb = two_step_baseline(h_ext, lp_ext, self.rho, self.sigma_sq,
                                   max_iter=self.ts_max_iter, tol=self.ts_tol)
            alpha = alpha_1b(lp, log_baseline_gaussian(h, tb["mu_hat"], self.sigma_sq), self.rho)
            kappa, bound, gap = tb["kappa"], tb["bound"], tb["gap"]
        if self.omit_duration_norm:
            R = (alpha.unsqueeze(1) * h).sum(dim=0)   # §13 row 8 ablation: no 1/T_b
        else:
            R = bag_statistic(alpha, h)               # Eq. 17: (1/T_b) Σ_t α_t h′_t
        return {"R": R, "alpha_mean": float(alpha.mean()), "alpha_std": float(alpha.std()),
                "T_b": int(h.shape[0]), "kappa": float(kappa), "bound": float(bound),
                "gap": float(gap)}

    @torch.no_grad()
    def encode_bags(self, z_list, labels=None, mode="per_bag", buffers=None):
        if len(z_list) == 0:
            raise ValueError("encode_bags: empty bag list (B14 discipline)")
        labels = self._check_labels(labels, len(z_list), self.num_classes)
        stats = []
        for i, z in enumerate(z_list):
            bz = buffers[i] if buffers is not None else None
            s = self.encode_bag(z, mode=mode, buffer_z=bz)
            if labels is not None:
                s["label"] = int(labels[i])
            stats.append(s)
        R = torch.stack([s["R"] for s in stats], dim=0)
        return R, stats

    # ------------------------------------------------------------------ classifier (§10.2/11.2)

    def fit_nearest_centroid(self, R, labels):
        labels = self._check_labels(labels, R.shape[0], self.num_classes)
        if labels is None:
            raise ValueError("fit_nearest_centroid requires labels")
        R = R.detach().float().to(self.rho.device)
        C = self.num_classes
        m = torch.zeros(C, R.shape[1], dtype=torch.float64)
        counts = torch.zeros(C, dtype=torch.long)
        for y in range(C):
            sel = (labels == y)
            if not bool(sel.any()):
                raise ValueError(f"class {y} has zero training bags — centroid/Eq. 18 head "
                                 "cannot be formed (no silent zero-centroid class)")
            m[y] = R[sel].double().mean(dim=0)
            counts[y] = int(sel.sum())
        resid2 = ((R.double() - m[labels.to(R.device)]) ** 2).sum(dim=1)
        sigma_sq = float(resid2.mean() / R.shape[1])
        if sigma_sq <= 1e-12:
            raise RuntimeError(f"sigma_R^2 = {sigma_sq:.3e} degenerate (R_b collapse?) — "
                               "Eq. 18 head needs positive variance, refusing warm-start")
        with torch.no_grad():
            self.centroids.copy_(m.to(self.centroids.dtype))
            self.sigma_R_sq.fill_(sigma_sq)
        self._fitted = True
        return {"sigma_R_sq": sigma_sq, "class_counts": counts.tolist()}

    def init_bayes_head(self, ablate_bias=False):
        """Eq. 18 warm-start; provably the same decision rule as nearest centroid.
        ablate_bias=True is the §13 row 9 Failure-Catalogue ablation (b_y ≡ 0):
        losing burst-rate discrimination is the EXPECTED outcome, not a bug."""
        if not self._fitted:
            raise RuntimeError("init_bayes_head before fit_nearest_centroid")
        if not isinstance(ablate_bias, bool):
            raise TypeError(f"ablate_bias must be bool, got {ablate_bias!r}")
        s2 = float(self.sigma_R_sq)
        head = nn.Linear(self.centroids.shape[1], self.num_classes, bias=True)
        with torch.no_grad():
            head.weight.data.copy_(self.centroids.double() / s2).float().to(self.centroids.device)
            head.bias.data.copy_(-(self.centroids.double() ** 2).sum(dim=1) / (2.0 * s2)
                                 + math.log(1.0 / self.num_classes)).float().to(self.centroids.device)
            if ablate_bias:
                print("[Phase1BReadout] ABLATION ON: b_y = 0 (§13 row 9) - largest-rate "
                      "class collapse is the EXPECTED result.")
                head.bias.data.zero_()
        self.head = head
        return head

    @torch.no_grad()
    def centroid_predict(self, R):
        if not self._fitted:
            raise RuntimeError("centroid_predict before fit_nearest_centroid")
        d = torch.cdist(R.float(), self.centroids.float())
        return d.argmin(dim=1)

    @torch.no_grad()
    def head_predict(self, R):
        if self.head is None:
            raise RuntimeError("head_predict before init_bayes_head")
        return self.head(R.float()).argmax(dim=1)

    @torch.no_grad()
    def evaluate(self, R, labels, use="head"):
        labels = self._check_labels(labels, R.shape[0], self.num_classes)
        pred = self.head_predict(R) if use == "head" else self.centroid_predict(R)
        return float((pred == labels.to(pred.device)).float().mean())

    def finetune(self, R, labels, epochs=30, lr=1e-3, lambda_wd=0.05):
        """CE + L2 weight decay on the SINGLE linear head (plan §5 Stage 7)."""
        if self.head is None:
            raise RuntimeError("finetune before init_bayes_head")
        if not (1e-2 <= lambda_wd <= 1e-1):
            raise ValueError(f"lambda_wd must lie in [1e-2, 1e-1] (approved range), got {lambda_wd}")
        labels = self._check_labels(labels, R.shape[0], self.num_classes)
        R = R.detach().float().to(self.centroids.device)
        y = labels.to(self.centroids.device)
        opt = torch.optim.AdamW(self.head.parameters(), lr=lr, weight_decay=lambda_wd)
        history = []
        for _ in range(int(epochs)):
            opt.zero_grad()
            loss = F.cross_entropy(self.head(R), y)
            loss.backward()
            opt.step()
            history.append({"loss": float(loss.detach()),
                            "acc": float((self.head(R).argmax(1) == y).float().mean())})
        return history

    # ------------------------------------------------------------------ head state I/O

    def head_state(self):
        if self.head is None:
            raise RuntimeError("head_state before init_bayes_head")
        return {"schema": HEAD_SCHEMA, "state_dict": {k: v.detach().cpu().clone()
                                                     for k, v in self.head.state_dict().items()},
                "sigma_R_sq": float(self.sigma_R_sq),
                "centroids": self.centroids.detach().cpu().clone()}

    def load_head_state(self, blob):
        if not isinstance(blob, dict) or blob.get("schema") != HEAD_SCHEMA:
            raise ValueError(f"load_head_state: schema must be {HEAD_SCHEMA!r}")
        if not self._fitted:
            raise RuntimeError("load_head_state: fit_nearest_centroid first (strict order)")
        if abs(float(blob["sigma_R_sq"]) - float(self.sigma_R_sq)) > 1e-6 * max(
                1.0, abs(float(self.sigma_R_sq))):
            raise ValueError("load_head_state: sigma_R_sq mismatch vs fitted geometry")
        if not torch.allclose(blob["centroids"].double(), self.centroids.double(), atol=1e-6):
            raise ValueError("load_head_state: centroids mismatch vs fitted geometry")
        self.init_bayes_head()
        missing = self.head.load_state_dict(blob["state_dict"], strict=True)
        assert not missing.missing_keys and not missing.unexpected_keys
        return self


def run_phase_1b(payload, train_z, train_labels, eval_z, eval_labels, *,
                 lambda_wd=0.05, finetune_epochs=30, lr=1e-3, sigma_sq=1.0,
                 mode="per_bag", buffers=None,
                 omit_duration_norm=False, ablate_bias=False):
    """Full Phase 1B pass in the mandated order; returns the smoke-report dict.
    finetune_epochs=0 skips fine-tuning (pure zero-train evaluation).
    omit_duration_norm / ablate_bias are the §13 Failure-Catalogue ablation
    switches (defaults False = spec behaviour)."""
    rd = Phase1BReadout(payload, sigma_sq=sigma_sq, omit_duration_norm=omit_duration_norm)
    Rtr, statstr = rd.encode_bags(train_z, train_labels, mode=mode,
                                  buffers=buffers.get("train") if isinstance(buffers, dict) else buffers)
    fit = rd.fit_nearest_centroid(Rtr, train_labels)
    rd.init_bayes_head(ablate_bias=ablate_bias)
    Rte, _ = rd.encode_bags(eval_z, eval_labels, mode=mode,
                            buffers=buffers.get("eval") if isinstance(buffers, dict) else None)
    zero_acc = rd.evaluate(Rte, eval_labels, use="centroid")
    ft_acc = None
    if finetune_epochs:
        rd.finetune(Rtr, train_labels, epochs=finetune_epochs, lr=lr, lambda_wd=lambda_wd)
        ft_acc = rd.evaluate(Rte, eval_labels, use="head")
    return {"zero_shot_acc": zero_acc, "finetuned_acc": ft_acc,
            "sigma_R_sq": fit["sigma_R_sq"], "class_counts": fit["class_counts"],
            "kappa_max": max(s["kappa"] for s in statstr),
            "bound_mean": sum(s["bound"] for s in statstr) / len(statstr),
            "n_train": Rtr.shape[0], "n_eval": Rte.shape[0], "mode": mode,
            "ablations": {"omit_duration_norm": bool(omit_duration_norm),
                          "ablate_bias": bool(ablate_bias)}}
