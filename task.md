Here is a compact, copy-paste prompt for your coding agent:

---

> Apply these 2 quick fixes and lock in the smoke-run config (`Subjects 3–7` train $\to$ `1–2` val, all 3 sessions):
> **Fixes:**
> 1. **`estep.py` (drop `(d - m)` divisor in $\hat{\sigma}^2$):** Compute total radial baseline spread instead of per-dimension variance:
> 
> $$\hat{\sigma}^2 = \max\!\left(\sigma_{\min}^2,\; \frac{\sum_t (1 - \alpha_t)\Vert{}h'_t - \hat{\mu}'_{s_t}\Vert{}^2}{\sum_t (1 - \alpha_t)}\right), \qquad \sigma_{\min}^2 = 0.05.$$
> 
> 
> 
> *(Prevents $\ell_s$ from drowning out $\ell_y$ by $50\times$ and keeps Proposition 10.2's $\kappa_\mu < 1$.)*
> 2. **`sinkhorn.py` / `sm_loss.py` (`eps = 0.15`, `sinkhorn_iters = 25`, log-domain):** Keep Sinkhorn in log-domain with `eps = 0.15` and `25` iterations. Downgrade the startup worst-case Birkhoff `eps_lower_bound` check to a warning (do not hard-abort on `eps = 0.15`) and log the empirical marginal residual `max(|Pi.sum(1)-a|, |Pi.sum(0)-b|)` (verify `< 1e-3`).
> 
> 
> **Smoke-Run Hyperparameters (`ea_supcon_fda_adamw_lotomasking.yaml`):**
> * **Split & Schedule:** `--loso 1,2` (train `3–7`, val `1–2`, all 3 sessions), `epochs: 25`, `batch_size: 64` (`PKSampler` eff `60`, `subject_diverse_k: false`), `probe_interval: 2`.
> * **Optimizer & AMP:** `AdamW`, `lr: 1e-3` (cosine $\to$ `1e-5`), `weight_decay: 0.01`, `bf16` autocast on backbone + momentum (no `GradScaler`), `fp32` in `SMCLossEngine`.
> * **Queue & Subspace:** `queue_size: 2048`, `allow_smoke_sizing: true`, MoCo momentum linear `0.90 -> 0.99` over `loss.sm.warmup_epochs: 2` (with `U = 0` during warmup), dynamic rank `0.85` trace capped at `m <= 3`, `tau_norm: 0.25`.
> * **Loss & Gate:** `lambda_anc: 1.0` (`tau_DCL: 0.1`, `anchor_on: projected`), `eps: 0.15`, `sinkhorn_iters: 25`, `tau_0: 0.35`, `kappa_cost: 0.50`, `c: 0.5`, `g_floor: 0.1`, `lambda_max: 1.0`, `lambda_sub: 0.1`, `lambda_rho: 0.1`.
> * **Calibrated E-Step:** `T: 2.0 -> 1.0` over first 30% of steps, `bisection_iters: 25`, `u_y` EMA `0.95`, `sigma_min^2: 0.05`, `rho_min: 0.05`.
> * **Augmentations:** `FourierMixup3D` on `query` only (`p: 0.5`, `retention_ratio: 0.95`, `Beta(1,1)`); **`key` view `p: 0.0**`.
> 
> 
> Confirm tests pass (including `marginal_residual < 1e-3` at `eps = 0.15`) so I can pull and launch on RunPod.

---

## Phase-1B head: LBFGS (2026-11-26, fifth) — APPLIED

`Phase1BReadout.finetune` fits the (256→5) head with full-batch **LBFGS** (strong Wolfe, history 10; `epochs`→max_iter, `lr`→step scale, default 1.0 via `run_phase_1b(lr=None)`) straight off the Eq. 18 (w, b) — convex, frozen features, ~1.3k params, so quasi-Newton is the right tool and it converged in 3 closure evals on the stage-7 data (loss 0.00169 → 0.00164, acc unchanged). One deliberate deviation: the λ_wd penalty is **proximal** (`λ‖(W,b)−(W₀,b₀)‖²` around the Bayes init), not a zero-centered ridge — LBFGS actually minimizes what it's given, and a coupled ridge on W would shrink the 1/σ_R² Bayes scale while b stays fixed (measured pure CE 0.0017 → 1.02, decision-equivalence structure broken). The old AdamW loop's decoupled decay was numerically ~0 over 30 steps anyway. Battery **121, 120 green**; stage7 pins `epochs=0`/`lr=0` raises + strict descent from the warm start.

---

## Option-B seed-stat fix (2026-11-25, fourth) — APPLIED

Confirmed by measurement: the pre-pass summed `d_t^2` over the ENTIRE window (T·C·H·W = 163,840 entries) with `B = 256` — median Δ ≈ 7.4e4, 100% sigmoid saturation, every class seeding at the 1−rho_min clamp (garbage priors). Now (train.py pre-pass): `r_t = x.mean(dim=(T,H,W))` → (5,) band response → corpus per-band median + **σ = IQR/1.349** (CLT makes the pooled mean Gaussian, so this is the consistent σ; owner call 2026-11-25) → `d_t^2 = ||(r−med)/σ||^2` → `seed_priors(..., log_coeff=5)`. Synthetic check: ρ = 0.116 vs true burst rate 0.106 (old path: 0.95 flat). `seed_priors` gained the explicit `log_coeff` argument (0/NaN rejected); `generate_dataset.py` untouched (its med/IQR is the on-disk wavelet normalization — different stage). Pin: `test_seed_priors_log_coeff_override`. Battery **121, 120 green**.

## Run commands (paths patched in, 2026-11-25)

Pod paths: SEED-V raw + `coords.csv` at `/workspace/SEED-V`, preprocessed subject 1–7 slice (all 3 sessions) at `/workspace/SEED-V/preprocessed_eeg_ea_1_7`. The slice is already windowed (fp16), so no `--setup_data` and — since masks are optional now — no `--masks` step.

```bash
cd /workspace/eeg_analysis_snn

python main.py --config config/ea_supcon_fda_adamw_lotomasking.yaml --loso 1,2 \
  data.dataset_path=/workspace/SEED-V/preprocessed_eeg_ea_1_7
```

Everything else comes from the locked config (25 epochs, B=64 → PK eff 60, `n_q=2048`, eps 0.15 / 25 iters / residual_cap 1e-3, anneal 2.0→1.0 over 30%, `rho_min 0.05`, `max_rank 3`, warmup 2, `lr_min 1e-5`).

> **Parser fix (2026-11-25, pull before launching):** the `data.dataset_path=...` override used to die with `main.py: error: unrecognized arguments` — argparse ran strict (`parse_args`) before the Hydra-forwarding block could see it. `main.py` now uses `parse_known_args()`: trailing `key=value` (or `--key=value`) tokens become Hydra overrides; anything else still errors with a usage hint. Verified: the override reaches the composed config, and the `--no_train` dry exit works.

> **Batch-unpack fix (2026-11-26, second pod crash — pull again):** `AttributeError: 'list' object has no attribute 'to'` at `train.py:391 timestamps = timestamps.to(device)`. Cause: `dataset.__getitem__` returned slot 7 (window_ids) as a Python list and PyTorch's default collate turns a batch of equal-size lists into a list of per-position tensors. Fix: dataset now returns `torch.tensor(window_ids, dtype=torch.long)` in both tuple paths; the loop's existing `timestamps.dim()/[:,0]` consumers are unchanged. Verified locally through the exact train.py unpack lines; pinned in stage 0.

Watch at startup: `[SMC §12.2 WARNING] eps=0.1500 < B2a worst-case floor 0.9233` (expected, it fires per class), then §9.6 sizing/staleness lines. If the §9.4 staleness gate raises, the queue is too long for the actual `steps_per_epoch` — halve `loss.sm.n_q` and relaunch (it hard-raises by design, never warns). During training `smc/ot_residual` must stay < 1e-3; a class whose rho drifts past ~0.5 can breach at 25 iters — the ladder answer is more `sinkhorn.n_iters`, not a bigger cap.

## M-step ticket (2026-11-25, third) — APPLIED

- **`detach(alpha)` everywhere in the M-step**: it was already detached at the rho M-step, L_sub, burst marginal, σ̂² weights, enqueue betas and diagnostics — the one live grad path was **L_spr**, now `sparsity_loss(alpha.detach(), ...)`. Encoder gradients come only through z~ in L_anc/L_em/L_sub.
- **Raw keys in the queue**: the ring stores the raw momentum embeddings on S^255; `NullspaceState.project` with the CURRENT (mu, U) runs on every READ (clouds, DCL negatives, lockpass payload). Enqueue-time projected keys can no longer go stale in the subspace. New `key_contract` buffer makes today's pre-ticket queue checkpoints fail the strict load.
- **SVD / lambda_OT detach**: the real bug this item pointed at — `apply_projector` fed a *view* of the U buffer into the autograd graph while `_recompute_geometry` zeroes U in-place on the next M-step → `RuntimeError: modified by an inplace operation` on the first backward with rank>0 (anomaly-verified; would have killed the pod run at epoch 2). U is now CLONED for the graph; SVD input gets an explicit `.detach()`; `lambda_OT` was already a plain float (pinned in the test).
- **Step-1 empty-slice guards**: (a) if the ring holds <2 distinct trials and all sit inside the batch, the anchor is mathematically inapplicable → `cold_anchor + anchor_coldstart` flag instead of the DCL "every query lost ALL negatives" raise (the tripwire still fires for any real queue); (b) `class_ot_term` fully-forbidden cross blocks now ship the EXACT degenerate plan (d=a, u=b, value 2τ+entropy, residual 0) instead of stalling the log-domain iteration on all-inf rows (measured 6.2e-3 residual at 800 iters).
- **2 log scalars**: `Phase1A/SMC/SigmaHatSq` (calibrated radial spread — is l_s on an informative scale) and `Phase1A/SMC/OTResidualMax` (worst empirical marginal residual — the eps=0.15/25-iter contract, live every epoch).
- Tests: **120, 119 green** (pre-existing `run_phase1b_t0_t5` import only). New pin `test_mstep_detach_raw_keys_and_step1_guards` covers the detach-grads, raw storage, both step-1 guards and the backward-after-geometry-mutation regression. **Voids: today's queue checkpoints** (contract marker + raw semantics) — fresh start on the pod.

## Done (applied 2026-11-25)

- **Fix 1:** `sigma_hat^2` is now the total radial spread (no `(d−m)` divisor), floor `sigma_sq` default **0.05** (`sm_loss.py` + config + docs).
- **Fix 2:** `eps: 0.15`, `sinkhorn.n_iters: 25`, start gate downgraded to warning (degenerate eps where tol is provably unreachable still hard-raises); empirical residual logged per step as `diagnostics["ot_residual"]`, accepted under new `sinkhorn.residual_cap: 1.0e-3`, breach still raises.
- Smoke hyperparameters locked in `config/ea_supcon_fda_adamw_lotomasking.yaml` (+ new `loss.sm.max_rank: 3` → `NullspaceState`, `training.lr_min: 1e-5` → scheduler).
- Tests: **119, 118 green** (only the pre-existing `run_phase1b_t0_t5` import fails). New pins: seeded-τ residual < 1e-3 at 25 iters in fp64, high-τ corners raise-and-resolve-at-1000, seed warns-but-runs at pinned 0.15, `min_iters_for_tol` degenerate raise kept, rank-cap test, `sigma_min² = 0.05` default.