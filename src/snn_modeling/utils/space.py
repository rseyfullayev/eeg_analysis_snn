"""Shared nullspace geometry & subject state for the SMC objective.

Implements fixed.pdf §2 (unified B^d geometry), §6.3-6.6 (subject M-steps),
Rm 6.4 (EMA tables + per-step thin SVD refresh), Rm 6.5 (zero-shot unseen-
subject annihilation), Rm 2.1 (norm-floored ball features), §11.4(ii) (dynamic
rank band + U=0 warmup).

EVERY consumer — projected anchor (Stage 5), dustbin OT clouds (Stage 3/4),
E-step densities (Stage 3), Phase-1B R_b readout (Stage 7) — gets h' and z-tilde
ONLY through project(). That single path is what makes the Failure-Catalogue
"split 128-D queue vs 256-D readout" row unrepresentable by construction.
"""

import torch
import torch.nn as nn

LOCK_SCHEMA = "smc-space-v1"


class NullspaceState(nn.Module):
    """State object owning (mu, U, m, tau_norm, EMA subject tables, epoch)."""

    def __init__(self, feature_dim=256, num_subjects=15, tau_norm=0.25,
                 ema_momentum=0.9, rank_energy_band=0.85, warmup_epochs=2):
        super().__init__()
        if feature_dim < 2:
            raise ValueError(f"feature_dim must be >= 2, got {feature_dim}")
        if num_subjects < 2:
            raise ValueError(
                f"num_subjects must be >= 2 (Remark 6.4: centered subject means span <= S-1), got {num_subjects}")
        if not 0.0 < tau_norm:
            raise ValueError(f"tau_norm must be > 0, got {tau_norm}")
        if not 0.0 <= ema_momentum < 1.0:
            raise ValueError(f"ema_momentum must be in [0, 1), got {ema_momentum}")
        if not 0.0 < rank_energy_band <= 1.0:
            raise ValueError(f"rank_energy_band must be in (0, 1], got {rank_energy_band}")
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")

        self.feature_dim = feature_dim
        self.num_subjects = num_subjects
        self.tau_norm = tau_norm
        self.ema_momentum = ema_momentum
        self.rank_energy_band = rank_energy_band
        self.warmup_epochs = warmup_epochs
        self.max_rank = num_subjects - 1

        self.register_buffer("mu", torch.zeros(feature_dim))
        self.register_buffer("U", torch.zeros(feature_dim, self.max_rank))
        self.register_buffer("rank", torch.zeros(1, dtype=torch.long))
        self.register_buffer("ema_mean", torch.zeros(num_subjects, feature_dim))
        self.register_buffer("ema_mass", torch.zeros(num_subjects))
        self.register_buffer("has_data", torch.zeros(num_subjects, dtype=torch.bool))
        self.register_buffer("epoch", torch.zeros(1, dtype=torch.long))

    # ------------------------------------------------------------------ lifecycle

    @property
    def warmup_active(self) -> bool:
        return int(self.epoch.item()) < self.warmup_epochs

    def advance_epoch(self):
        self.epoch += 1

    def set_epoch(self, epoch: int):
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError(f"epoch must be >= 0, got {epoch}")
        self.epoch.fill_(epoch)

    # ------------------------------------------------------------------ geometry

    def apply_projector(self, h: torch.Tensor) -> torch.Tensor:
        """h - U U^T h (Rm 6.5: exact for any h containing an in-span subject
        offset). During warmup or rank 0 this is the identity (§11.4(ii))."""
        if h.dim() != 2 or h.shape[1] != self.feature_dim:
            raise ValueError(f"apply_projector expects (B, {self.feature_dim}), got {tuple(h.shape)}")
        if not torch.isfinite(h).all():
            raise ValueError("apply_projector received non-finite values")
        r = int(self.rank.item())
        if self.warmup_active or r == 0:
            return h
        V = self.U[:, :r]
        return h - (h @ V) @ V.t()

    def project(self, z: torch.Tensor):
        """(B, D) unit-norm z (fixed.pdf §2) -> (h', z_tilde, h_norm).

        h' = (I - UU^T)(z - mu); z_tilde = h' / max(||h'||, tau_norm) ∈ B^d.
        """
        if z.dim() != 2 or z.shape[1] != self.feature_dim:
            raise ValueError(f"project expects (B, {self.feature_dim}), got {tuple(z.shape)}")
        if not torch.isfinite(z).all():
            raise ValueError("project received non-finite z")
        norms = z.norm(dim=1)
        if (norms - 1.0).abs().max() > 1e-3:
            raise ValueError(
                f"project expects L2-normalized z on S^(d-1) (fixed.pdf §2); "
                f"row norms span [{norms.min():.4f}, {norms.max():.4f}]")
        h = z - self.mu
        h_prime = self.apply_projector(h)
        h_norm = h_prime.norm(dim=1)
        denom = torch.clamp(h_norm, min=self.tau_norm).unsqueeze(1)
        z_tilde = h_prime / denom
        if (z_tilde.norm(dim=1) > 1.0 + 1e-5).any():
            raise RuntimeError("norm floor violated: |z_tilde| > 1 — this is a bug in space.py")
        return h_prime, z_tilde, h_norm

    # ------------------------------------------------------------------ M-steps

    @torch.no_grad()
    def update_batch(self, z: torch.Tensor, alpha: torch.Tensor, subject: torch.Tensor):
        """One EM mini-batch M-step (Prop 6.2 + Rm 6.4).

        z: (B, D) unit-norm online embeddings; alpha: (B,) E-step burst
        responsibilities (Rm 4.2 interiority keeps them in (0,1)); subject:
        (B,) long ids in [0, num_subjects). Baseline-weighted means/masses are
        EMA-tracked per subject; (mu, U, rank) are recomputed from the tables —
        all under no_grad: U is an M-step quantity, gradients reach it only
        through L_sub(z) (plan §5 Stage 1).
        """
        if z.dim() != 2 or z.shape[1] != self.feature_dim:
            raise ValueError(f"update_batch z must be (B, {self.feature_dim}), got {tuple(z.shape)}")
        B = z.shape[0]
        if alpha.shape != (B,):
            raise ValueError(f"alpha must be (B,)={B}, got {tuple(alpha.shape)}")
        if subject.shape != (B,):
            raise ValueError(f"subject must be (B,)={B}, got {tuple(subject.shape)}")
        if not torch.isfinite(z).all() or not torch.isfinite(alpha).all():
            raise ValueError("update_batch received non-finite z/alpha")
        if alpha.min() < 0.0 or alpha.max() > 1.0:
            raise ValueError(f"alpha must lie in [0, 1], got [{alpha.min():.4f}, {alpha.max():.4f}]")
        if subject.dtype != torch.long:
            raise ValueError(f"subject must be long tensor, got {subject.dtype}")
        if (subject < 0).any() or subject.max() >= self.num_subjects:
            raise ValueError(
                f"subject ids must lie in [0, {self.num_subjects}) — the old -1 sentinel is "
                "no longer tolerated (audit B7); fix the dataset mapping")
        norms = z.norm(dim=1)
        if (norms - 1.0).abs().max() > 1e-3:
            raise ValueError("update_batch expects L2-normalized z (fixed.pdf §2)")

        w = 1.0 - alpha  # baseline weight (Prop 6.2)
        m = self.ema_momentum
        for s in subject.unique().tolist():
            mask = subject == s
            ws = w[mask]
            mass = float(ws.sum().item())
            if mass <= 0.0:
                continue  # subject fully burst-flagged this batch: nothing to update
            zs = z[mask]
            mean = (ws.unsqueeze(1) * zs).sum(dim=0) / mass
            if bool(self.has_data[s]):
                self.ema_mean[s].mul_(m).add_(mean, alpha=1.0 - m)
                self.ema_mass[s].mul_(m).add_(torch.as_tensor(mass, device=self.ema_mass.device,
                                                              dtype=self.ema_mass.dtype),
                                              alpha=1.0 - m)
            else:
                self.ema_mean[s].copy_(mean)
                self.ema_mass[s].fill_(mass)
                self.has_data[s] = True
        self._recompute_geometry()

    @torch.no_grad()
    def _recompute_geometry(self):
        present = self.has_data & (self.ema_mass > 0)
        n_present = int(present.sum().item())
        if n_present == 0:
            self.mu.zero_()
            self.U.zero_()
            self.rank.zero_()
            return
        N = self.ema_mass[present]                       # (S_p,)
        mu = (N.unsqueeze(1) * self.ema_mean[present]).sum(dim=0) / N.sum()
        self.mu.copy_(mu)
        self.U.zero_()
        if n_present < 2:
            self.rank.zero_()
            return
        Vs = self.ema_mean[present] - mu                 # (S_p, D) centered means
        M = (Vs * (N / N.sum()).sqrt().unsqueeze(1)).t()  # (D, S_p): M M^T == S_between
        Uv, Sv, _ = torch.linalg.svd(M, full_matrices=False)
        energy = Sv ** 2
        total = float(energy.sum().item())
        if total <= 0.0:
            self.rank.zero_()
            return
        cum = torch.cumsum(energy, dim=0) / total
        r = int(torch.searchsorted(cum.double(),
                                   torch.tensor(self.rank_energy_band, dtype=torch.double)).item()) + 1
        r = max(0, min(r, self.max_rank, n_present - 1))  # Remark 6.4: rank <= S-1
        self.U[:, :r] = Uv[:, :r]
        self.rank.fill_(r)

    # ------------------------------------------------------------------ diagnostics

    def summary(self) -> dict:
        r = int(self.rank.item())
        out = {
            "epoch": int(self.epoch.item()),
            "warmup_active": self.warmup_active,
            "rank": r,
            "subjects_with_data": int(self.has_data.sum().item()),
            "tau_norm": self.tau_norm,
            "rank_energy_band": self.rank_energy_band,
        }
        if r > 0:
            out["explained_energy_ratio"] = float(self._explained_energy().item())
        return out

    @torch.no_grad()
    def _explained_energy(self) -> torch.Tensor:
        r = int(self.rank.item())
        present = self.has_data & (self.ema_mass > 0)
        if r == 0 or int(present.sum().item()) < 2:
            return torch.zeros(())
        N = self.ema_mass[present]
        Vs = self.ema_mean[present] - self.mu
        total = float(((N.unsqueeze(1) * Vs ** 2).sum()).item())
        V = self.U[:, :r]
        captured = float(((N.unsqueeze(1) * (Vs @ V) ** 2).sum()).item())
        return torch.tensor(captured / max(total, 1e-12))

    # ------------------------------------------------------------------ lock I/O

    def lock_payload(self) -> dict:
        """Exact (mu, U, rank) lock written at the end of Phase 1A (Rm 6.4)."""
        r = int(self.rank.item())
        return {
            "schema": LOCK_SCHEMA,
            "feature_dim": self.feature_dim,
            "num_subjects": self.num_subjects,
            "tau_norm": float(self.tau_norm),
            "ema_momentum": float(self.ema_momentum),
            "rank_energy_band": float(self.rank_energy_band),
            "warmup_epochs": int(self.warmup_epochs),
            "epoch": int(self.epoch.item()),
            "rank": r,
            "mu": self.mu.detach().cpu().clone(),
            "U": self.U[:, :r].detach().cpu().clone().contiguous(),
        }

    def load_lock(self, payload: dict):
        """Strictly validate + install a lock payload. Any mismatch raises."""
        if not isinstance(payload, dict) or payload.get("schema") != LOCK_SCHEMA:
            raise ValueError(
                f"lock payload schema must be {LOCK_SCHEMA!r}, got "
                f"{payload.get('schema') if isinstance(payload, dict) else type(payload)}")
        for key in ("feature_dim", "num_subjects", "tau_norm"):
            if key not in payload:
                raise ValueError(f"lock payload missing {key!r}")
        if payload["feature_dim"] != self.feature_dim or payload["num_subjects"] != self.num_subjects:
            raise ValueError(
                f"lock geometry mismatch: payload (d={payload['feature_dim']}, "
                f"S={payload['num_subjects']}) vs state (d={self.feature_dim}, S={self.num_subjects})")
        r = int(payload["rank"])
        if not 0 <= r <= self.max_rank:
            raise ValueError(f"lock rank {r} outside [0, {self.max_rank}]")
        mu, U = payload["mu"], payload["U"]
        if mu.shape != (self.feature_dim,) or not torch.isfinite(mu).all():
            raise ValueError(f"lock mu must be finite ({self.feature_dim},), got {tuple(mu.shape)}")
        if U.shape != (self.feature_dim, r) or (r > 0 and not torch.isfinite(U).all()):
            raise ValueError(f"lock U must be finite ({self.feature_dim}, {r}), got {tuple(U.shape)}")
        if r > 0:
            gram = U.t() @ U
            if not torch.allclose(gram, torch.eye(r, dtype=gram.dtype), atol=1e-4):
                raise ValueError("lock U columns are not orthonormal — refusing a corrupted artifact")
        with torch.no_grad():
            self.mu.copy_(mu.to(self.mu.device))
            self.U.zero_()
            if r > 0:
                self.U[:, :r] = U.to(self.U.device)
            self.rank.fill_(r)
            self.epoch.fill_(max(int(payload.get("epoch", self.warmup_epochs)), self.warmup_epochs))
        return self
