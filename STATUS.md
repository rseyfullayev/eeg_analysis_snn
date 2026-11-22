# SWEEP-Net — Project Status

*Companion to `REVIEW_SMC_AUDIT.md` (full plan + build log) and `CERTIFICATION_RUNBOOK.md` (lab-box run order). Authority: `fixed.pdf`. Snapshot 2026-10-21; test counts are live `def test_` counts of `tests/test_stage*.py`.*

## Build — Stages 0–8 (all delivered)

| Stage | Delivered | Tests | Status |
|---|---|---|---|
| **0 · Repair** | B1 permute fix + v2 bag payload (legacy artifacts refused at load); B2 runner rewritten as diagnostic harness; 25 silent-fallback sites purged; dead-code purge (DANN/VIB/RTFM/backup copies); checkpoint/CI/config fixes | 10/10 | ✅ 2026-10-19 |
| **1 · Geometry** (`utils/space.py`) | Unified d=256 nullspace on B^d: `project(z)` as the **single** geometry path; per-step EMA subject tables + thin 256×S SVD; dynamic rank (0.85 energy band); warmup; strict lock/lockpass save-load; Rm 6.5 unseen-subject annihilation | 9/9 | ✅ 2026-10-19 |
| **2 · Queue v2** (`memory/queue.py`) | Ring of (z̃, y, β=α detached, global uid, step); two reads — β-weighted class clouds + uniform negatives; same-window masking at both; ESS_y / D_norm_y; §9.6 sizing + §9.4 staleness init asserts | 9/9 | ✅ 2026-10-19 |
| **3 · E-step & sparsity** (`utils/estep.py`, `sparsity.py`) | Option-B ρ seeding (Banach iteration, interiority box); α^(1A) with annealed T 20→1; label-free α^(1B) + per-bag certified G² baseline; L_spr = bkl(α, ρ_y) + ρ self-consistency | 11/11 | ✅ 2026-10-19 |
| **4 · OT** (`utils/sinkhorn.py`) | Balanced dustbin Sinkhorn on augmented (n+1)×(m+1) kernel; τ_y = −ε·log(1−ρ_y); debiased S_ε; Danskin stop-grad coupling (never unrolled); B2a ε-interval guard | 13/13 | ✅ 2026-10-19 |
| **5 · Anchor + gates** (`utils/dcl.py`, `gates.py`) | DCL on z̃ (τ=0.1) exposing certificate C_K; readiness gate g = clamp(C_K/log(K−1), 0.1, 1); λ_OT = c·grad-ratio·g with EMA; B5(iii) c-band ctor assert | 14/14 | ✅ 2026-10-19 |
| **6 · Assembled objective** (`utils/sm_loss.py`) | Eq. 23 `SMCLossEngine` + full Phase 1A rewire: every knob under `loss.sm` (config-exposed, none hardcoded), every §12 inequality an init assert, bootstrap order tested, lockpass written at 1A end | 12/12 | ✅ 2026-10-19 |
| **7 · Phase 1B** (`adapters/mil_readout.py`) | R_b = (1/T_b)·Σ α^(1B) h′ from strict lockpass; zero-train nearest centroid → single Bayes-warm-started `nn.Linear(256,C)` head (Eq. 18); λ_wd ∈ [1e−2, 1e−1]; rolling-buffer variant behind flag; subject field never read | 6/6 | ✅ 2026-10-19 |
| **8 · Certification** (`utils/certification.py`, `CERTIFICATION_RUNBOOK.md`, `train.py`) | Table-1 band grading + Remark 11.2 smoke gate + Remark 11.1 remediation ladder (strict causal order) + B3(iii) α-identity early-stop tripwire + 11-row §13 Failure-Catalogue registry with loud ablation switches | 7/7 | ✅ built 2026-10-19 — **runs pending on lab box** |

**Suite total: 91/91 green** — full battery re-run 2026-10-21, all nine files pass (plain-script runner; no pytest in venv). *(The audit §8 trailer previously summed "74" — arithmetic slip, corrected there to 91.)* **Superseded 2026-11-20 by the pooling-head redesign below → 94/94.**

## Post-plan change · 2026-11-21 — live certification diagnostics + multi-subject validation

| What | Detail |
|---|---|
| `Val/Subject_Probe_hprime_Acc` | live h′-space subject probe (same SVM on `space.project(z)[0]`), logged next to the z-space one. Owner point adopted: z staying >70% is BY DESIGN (L_sub parks identity in span(U)); h′ is the number that must collapse <0.20 |
| `Val/ZeroStep_Bag_Acc` (+ `_Unweighted`) | live Phase-1B preview every probe: `Phase1BReadout(engine.lock_fullpass())` → R_b bags → nearest-centroid train→held-out; unweighted control = plain mean h′ (α≡1). Fit capped to first 256 train bags (announced); not-ready clouds → announced skip, never a fake number |
| `SMC/Rho_y{0..4}` | every epoch, next to DNorm_y — visual ρ ordering check (Fear/Disgust < Neutral) |
| Probe path fix | `train.py::_extract_feats` had its own GAP copy — probes were grading PRE-redesign geometry; now routes through `model.extract_features` (shared pool_head path) |
| Multi-subject validation | `--loso 1,2` (or `data.loso: [1,2]`): dataset/train, test, masks (`generate_masks` isin-fixed), FourierMixup cache tag (`loso1-2`) all list-aware; single int unchanged (verified) |
| `probe_interval` | was hardcoded 5 → config knob `training.probe_interval` (yaml: 5; smoke set 1 for every-epoch) |
| Seed key bug fixed | yaml `seed_batches` was READ BY NOBODY (code wants `seed_max_batches`) — renamed, knob now live |
| `requirements.lock.txt` | repo root — full `pip freeze` (Python 3.14.5, torch 2.9.1+cu130) for RunPod replication |
| Tests | battery 94/94 after the train.py probe edits; multi-loso + yaml keys verified by throwaway probe (deleted); final knob edits compile-checked |

## Post-plan change · 2026-11-20 — pooling-head redesign (owner spec)

| What | Detail |
|---|---|
| Backbone | stage4 `stride 2→1, dilation 4→2` — final conv map is now **(T,B,256,4,4)** (was 2×2); total downsampling 32→4 |
| Head | NEW `MeanMaxPoolHead` (`layers/stem.py`): depthwise 4×4 pad-0 conv (conv **is** the spatial squeeze; exact-fit guard raises on any other map size) → GroupNorm (per frame-sample only) → per-channel `0.5·mean_t + 0.5·max_t` → (B,256) unnormalized → caller's L2 stays the last step. No activation pre-pool (SiLU would rectify the negative half of the sphere) |
| Why | mean over 32 frames dilutes short α-bursts; the max term retains the peak (measured: burst channel 8.98 vs 2.00 mean-only) |
| Cost | +4,608 params (4,096 dw + 512 GN) → online total **828,677** (measured); momentum copy inherits via deepcopy |
| Shared path | `features.encode_windows`/`encode_bag_batch` now REQUIRE the model's `pool_head` (precompute + t0 harness load it strictly under `pool_head.*`) — payloads can never disagree with the trained embedding (B1-class skew unrepresentable) |
| Voids | every pre-change checkpoint/lockpass/`bag_*.pt` — old ckpts fail **strict load** on the new `pool_head.*` keys; full 1A retrain required (free: no lab numbers exist yet) |
| Tests | +3 in stage0 (blend formula exact, burst retention, guards; schedule pin: stage4==4×4 end-to-end; required-kwarg pin) → suite total **94** (13/9/9/11/13/14/12/6/7, battery 2026-11-20) |

## Post-plan change · 2026-11-22/23 — precision policy, cloud LOTO mask, asymmetric dual view

| What | Detail |
|---|---|
| On-disk dtype | `generate_dataset.py:210–215`: windows now `nan_to_num → clamp(−4,4) → float16`, with a loud tripwire if the range guard ever trips (upstream `log1p` + median/IQR already bounds them, so ±4 is exact in fp16). EA (`eigh`) and CWT power stay fp32/complex64. Load path unchanged (`dataset.py:344 .float()`) |
| AMP split | `train.py:428–450`: `torch.autocast(bfloat16)` wraps **only** encoder + classifier + momentum-key forward; on exit everything is cast to fp32 before `F.normalize(backbone_feats.float(), p=2, dim=1)` and the whole `SMCLossEngine` (SVD, DCL `logsumexp`, Sinkhorn `exp(−C/ε)`, bkl, queue, lockpass). **No GradScaler** — bf16 has fp32's exponent range, fp32 master weights update directly. `training.bf16: true` |
| FM3D cache staleness | `index.csv` is **byte-identical** after an fp32→fp16 regeneration, so `md5(index.csv)` alone would have silently reused the fp32-era donor centroids/σ. Cache key now also carries `_window_data_fingerprint` (per-subject representative window files keyed by `(subject, name, on-disk size)`) — the size term halves on a dtype switch |
| MoCo EMA ramp | 0.9 → 0.99 linear over `loss.sm.warmup_epochs` (= the §11.4ii U=0 window), resume-honest, logged as `Train/EMA_Momentum` + `Train/Amp_BF16`. Keys: `training.supmoco_momentum` (start), `loss.sm.supmoco_momentum_end` (target) |
| §9.6 floor re-based | `run_sizing_checks` now uses the **seeded** `min_y(π_y·ρ_y)` from the Option-B pre-pass instead of the `rho_min` guess; prints `expected burst keys/class (N_q·π_y·ρ_y)`. `allow_smoke_sizing: true` downgrades a sub-**floor** breach to a warning only — ceiling + §9.4 staleness still hard-raise. `sizing_d_eff` 64 → 20 |
| **Cloud LOTO support mask** (NEW) | Per-pair same-**trial** mask as a **structural zero** (`-inf` on `logK`), shared by the E-step `log p_y` and the OT cross term (Rm 4.2). A `C_ij = 1e4` sentinel is impossible here: `_validate_cost` raises outside [0,2] and `estep` clamps to 2.0, and it would push `eps_lower_bound` 0.92 → 4617. Dustbin row/col never maskable; self-terms unmasked. `loss.sm.ot_loto_mask: true`; logged as `SMC/LotoMaskOn`, `SMC/LotoForbiddenFrac`, `SMC/LotoColdRows`. Same-**subject** masking deliberately rejected — it routes freed mass to the dustbin and inflates `D_norm_y` (Table 1 #5) |
| **Asymmetric dual view** (fix) | `dataset.py` had ONE chain for both views → the momentum **key** view was FM3D-mixed, so donor spectra entered `ν_y` and `μ_s`. Now `augmentations` (query: FM3D + light) and `key_augmentations` (light, no FM3D) are separate; `training.key_view: light\|clean`. Two fail-loud guards: undeclared key view raises, a key chain containing FourierMixup raises (nested check). Probe's `clean_train_set` clears both |
| Tests | 94 → **105** (0:16 1:9 2:9 3:11 4:17 5:14 6:16 7:6 8:7) — **104 green** 2026-11-23; the one failure is the pre-existing `run_phase1b_t0_t5` import in stage 0 (that harness was deleted 2026-11-21; pending owner call) |

## Final audit (10 checks, `AUDIT.md`) — 2026-11-23

7 PASS / 3 DIFF at the time of audit; both actionable DIFFs are now closed above. Verified by measurement, not by reading comments: stage-4 map `(32,2,256,4,4)`, `extract_features → (B,256)`, no SiLU inside `pool_head`.

| Check | Verdict |
|---|---|
| 1 stage4 + tail, 2 single 256-D, 3 mixed precision, 5 nullspace-only E-step, 6 projected anchor + grad ratio, 7 drop price + full entropy, 8 masking consistency | **PASS** |
| 4 FM3D key-view leak | **DIFF → FIXED** (asymmetric dual view) |
| 9 warmup + sizing | **PASS**, but the key is `loss.sm.warmup_epochs`, not `loss.smc.*` |
| 10 R_b / G² / `Linear(256,5)` | **PASS**; probe cadence DIFF → set `training.probe_interval: 1` for every-epoch `h′`/ZeroStep logging (`SMC/Rho_y*` already logs every epoch) |

**Smoke-run config deltas** (yaml still carries the full-LOSO profile): `loss.sm.n_q=1024`, `loss.sm.allow_smoke_sizing=true`, `training.probe_interval=1`, `loss.sm.num_subjects=7`, `training.subject_diverse_k=false`.

## Findings ledger (this session's analysis pass)

| Item | What was found | State / action |
|---|---|---|
| **FourierMixup3D σ-override** | `train.py:1087–1093`: an explicit `fourier_mixup.sigma` rebuilds the Gaussian blend mask as **2-D (H,W)** — silently dropping the temporal-frequency falloff the binary-searched 3-D mask has | **Flagged, unfixed.** Dormant: active config pins only `retention_ratio/p/beta_alpha`. Fix before ever pinning σ |
| **Stride / final feature map** | Measured spatial schedule 32→16→8→4→2 (one stride-2 halving per stage, first block only). Final conv map is **(T, B, 256, 2, 2)**, not the 1×1 implied in audit §3.1's B1 text; GAP over T,H,W → 256-D unchanged | **Analysis only — no code change.** Recorded here as the correction |
| **Encoder architecture analysis** | Measured: backbone 724,869 params; projection head 99,200; online total **824,069** trainable (+ identical frozen momentum copy ≈ 1.65 M resident); active run is **analog SiLU mode** (spiking = config swap); 128-D head vestigial in the SMC path; queue 4.19 MB fp32 | Delivered 2026-10-21 (copy-paste spec block); facts in project memory |
| **Smoke-test implementation vs execution** | Stage-8 harness (evaluators, ladder, tripwires, runbook) is built and unit-tested; the **actual** §11.2 smoke gate, 16-fold LOSO, and §13 ablation runs have not been executed | **Owner-side**, lab box, per `CERTIFICATION_RUNBOOK.md` §0→§4 |
| **Telegram token (B20)** | Committed plaintext token/chat_id | **Deferred by owner** until repo publication (2026-10-16) — scheduled, not open |
