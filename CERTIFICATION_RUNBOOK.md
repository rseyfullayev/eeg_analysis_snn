# SWEEP-Net Certification Runbook — §11.5 protocol (fixed.pdf authority)

Stage 8 deliverable. Everything here runs on the **lab box** (data + GPU); the
repo side ships the code, the thresholds, the ladder and the tripwires.
Authoritative spec: `fixed.pdf` §11.5 (Table 1 + Remark 11.1 ladder + Remark 11.2
smoke protocol), §13 (Failure Catalogue), §12 (Track B). Evaluators:
`src/snn_modeling/utils/certification.py` (pure functions — never grade a metric
that was not measured: missing keys RAISE).

---

## 0. Owner prerequisites (before ANY run)

1. Repoint lab `START` to **`config/active/ea_supcon_fda_adamw_lotomasking.yaml`**
   (the run root where `config.loss` resolves; it carries the full `loss.sm`
   block, `enabled: true`). The retired `ea_fda_vib_1b.yaml` fails loudly at the
   phase guard — do not use it.
2. **Regenerate v2 bag payloads** (`bag_*.pt`) with the Stage-0 precompute
   entry point. Legacy bare-tensor payloads are refused by `load_bag_payload`
   (schema `v2`, FEATURE_DIM=256) — Phase 1B cannot read the old artifacts.
3. Sanity the §9.6 sizing constants once: `loss.sm.sizing_d_eff: 64.0` and
   `loss.sm.sizing_tol: 0.5` are the chosen constants that make
   floor ≤ n_q=4096 ≤ ceiling coherent with the approved defaults. If the
   owner's reading of §9.6 differs, `run_sizing_checks` will refuse the run at
   init — change the constants, never `n_q` silently.

## 1. Smoke gate (Remark 11.2) — minutes, before any full training

Slice by subjects/sessions, not windows (fewer than 4 subjects leaves S_between
rank-deficient):

| Setting | Value | Where |
|---|---|---|
| Subjects | S = 5 train + 1 unseen | pre-slice the bag index to 6 subjects |
| Sessions | Session 1 only ⇒ 15 trials/subject = **75 training bags**, ~3,000 windows, 15 unseen test bags | pre-slice |
| Queue | `loss.sm.n_q: 512` | config override |
| Rank | target m = 3 (≤ S−1); keep `rank_energy_band: 0.85`, check `NullspaceRank` scalar | config + TB |
| Batch | B ∈ {64, 128} | training config |
| Budget | 15–20 epochs | training config |

Run (per held subject):

```
python main.py --phase=1a --loso=<held_subject_id>
```

### Pass criteria (all four, within the 15–20-epoch budget)

| # | Criterion | Threshold | Read from |
|---|---|---|---|
| 1a | Subject probe on z_t | **starts > 80%** (content exists) | `Phase1A/Train/Subject_Probe_Val_Acc` (z) — epoch 1–2 |
| 1b | Subject probe on h′_t | **drops < 30%** (chance = 1/S ≈ 17%) | offline harness: encode windows → `NullspaceState.project` → 5-fold linear probe on h′ |
| 2 | Intra-trial Stdt(α_t) | **> 0.15** | `Phase1A/SMC/StdIntraTrial` (engine diag `stdt_intra`) |
| 3 | Unseen-subject bag accuracy | **≥ 35–40%** | `adapters/mil_readout.run_phase_1b(...)["zero_shot_acc"]` on the held subject's 15 bags (lockpass_1a.pt + v2 payloads) |

Grade with `certification.evaluate_smoke({probe_z, probe_h, stdt_alpha,
unseen_bag_acc})` and **log the verdict** (`smoke_verdict.json`). Gate #1a–1b is
a pair: high-z/low-h′ proves the eraser removed real, subject-bearing variance.

### Automatic tripwires already live in the loop
- **B3(iii) early stop**: `train.py` raises if `stdt_intra < 0.02` for 3
  consecutive epochs after the anneal floor (α ≡ ρ, uninformative gate).
  The message points here.
- `evaluate_table1` also flags energy shrinkage (`EnergyRatio` scalar,
  `Phase1A/SMC/EnergyRatio`; warmup epochs read 1.0 by design — judge after
  the rank lifts).

## 2. Remediation ladder if the smoke gate fails (Remark 11.1 — STRICT causal order)

Trigger: persistent `D_norm_y > 0.5` (scalar `Phase1A/SMC/DNorm_y{y}`). Call
`certification.remediation_ladder(dnorm_y, eps, eps_min, eps_max, rho_y,
rho_min, ess_y, n_q, rho_tilde_min)` with the run's ACTUAL values (eps +
interval from Thm 12.2 at seeding, ESS from `diag["ess"]` / `queue.ess_by_class`)
and apply ONLY `next_step`. One change per re-run. The ladder refuses to fire
on a healthy D_norm and refuses a silent all-clear (step 4 = escalate).

1. **ε position** (ε near ε_min ⇒ exp(−C/ε) ≪ 1−ρ_y forces mass into the drop
   slack): shift toward the admissible-interval midpoint — `loss.sm.eps` pins it.
2. **ρ_y seed** (under-seeded near ρ_min ⇒ τ_y too small, dropping nearly free):
   re-verify the Option-B fixed point and the burst spread — `loss.sm.seed.s` (>1).
3. **ESS_y / queue** (ν_y collapsed onto stale high-β outliers): increase
   `loss.sm.n_q` toward the ceiling or raise `loss.sm.anneal.t_start`.

## 3. Full certification — 16-fold LOSO

After a green smoke gate, per fold:

```
python main.py --phase=1a --loso=<1..16>          # writes lockpass_1a.pt
# then the Phase 1B readout (frozen): run_phase_1b(train bags, held-out subject)
```

Targets (Table 1, `certification.evaluate_table1(record)` per final epoch):
h′ probe < 15–20% (≈1/S chance) with z probe 70–90%; zero-step bag acc ≥ 38%
once g opens; energy ratio ∈ [0.35, 0.75]; g ∈ [0.4, 0.9]; D_norm_y ∈ [0.1, 0.5]
for every class; ρ_Fear/ρ_Disgust < ρ_Neutral (`rho_order_pairs`). Aggregate:
report per-fold verdicts + the mean acc; per-fold ladder transcripts kept.

## 4. Ablation matrix — §13 Failure Catalogue (11 rows, doubles as paper evidence)

From `certification.FAILURE_CATALOGUE_ROWS` (registry name is the runbook key —
`ablation_switch()` rejects typo'd names so a mis-typed ablation can never run
as the healthy config):

| Row | Switch | Expected pathology |
|---|---|---|
| Remove L_anc | `loss.sm.lambda_anc=0.0` | constant-vector/nullspace collapse |
| Remove unbalanced dropping | `loss.sm.force_balanced=true` | SCDA subject mining (~87% probe) |
| Remove nullspace projection | `loss.sm.warmup_epochs >= epochs` (+ `lambda_sub=0.0`) | probe → 91% |
| Split 1A queue vs 1B readout | UNREPRESENTABLE (single `project()` path); sibling: `loss.sm.anchor_on=raw` | ID stashed in head nullspace |
| log p_s on full space | UNREPRESENTABLE (signature) | jitter falsely triggers α_t |
| Remove L_spr | `loss.sm.lambda_rho=0.0` | ρ unconstrained; Fear/Disgust gaslighting |
| Learned α-head | PATH REMOVED (retired flags raise) | calibration failure |
| Omit 1/T_b | `Phase1BReadout(..., omit_duration_norm=True)` | trial-duration bias |
| Omit b_y | `init_bayes_head(ablate_bias=True)` | largest-rate collapse |
| Backprop through Sinkhorn | UNREPRESENTABLE (no_grad solve + Danskin coupling) | NaN blowup |
| Queue-vs-queue OT | UNREPRESENTABLE (Prop 9.1 guard) | zero-gradient dead term |

Ablation runs are smoke-sized (S=5+1) unless the paper needs the full fold set.
The two adapter flags announce themselves on stderr ("ABLATION ON") and must be
recorded in the run's config digest — never mix ablated runs into headline rows.

## 5. B1 re-baseline

The only historical numbers that remain valid are the **old Phase-1A** ones
(Proxy-A 64.10%, LOTO 50.81/84.08) — different objective, quoted as prior work.
Every 1B number from the `ea_fda_vib_1b` era is retired (corrupted bag pipeline,
B1): regenerate v2 payloads, re-run §3, report only new numbers.

## 6. Track-B runtime guards — status

| Obligation | Guard | Where |
|---|---|---|
| B2a ε-admissibility (Thm 12.2) | refused at seeding, per class | `SMCLossEngine.seed_priors` → `check_eps_admissible` |
| B3(iii) α ≡ ρ escape | early-stop raise (patience 3) | `train.py` via `certification.alpha_identity_escape` |
| B5(iii) 0 < c_min ≤ c ≤ c_max < 1 | ctor reject | `gates.check_c_gate` in `SMCLossEngine` |
| B6(v) Option-B fixed point | measured-κ tripwire | `sparsity.option_b_seed` |
| B6(vi) G² baseline κ < 1 | raises per bag, never a silent extra step | `estep.two_step_baseline` |
| B7 boundedness/Lipschitz on B^d | grid-verified + fuzz | `tests/test_stage5.py` (gate 1/log(K−1)), `tests/test_stage8.py` |

## 7. First-run checklist (new lab box)

- [ ] `START` points at the loss.sm config (§0.1)
- [ ] v2 payloads regenerated (§0.2)
- [ ] Smoke gate §1 → verdict logged
- [ ] If fail: ladder §2, one change at a time, re-run
- [ ] 16-fold §3 only after a green gate
- [ ] Ablations §4 only as separate, clearly-labelled runs
