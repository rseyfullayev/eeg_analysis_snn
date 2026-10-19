"""Stage 8 — certification evaluators for the §11.5 protocol (fixed.pdf).

Pure functions (no torch, no state) that encode:
  * Table 1: Phase-1A epoch-level diagnostic metrics + pass/fail bands,
  * Remark 11.2: the fast 5-subject smoke-test pass criteria,
  * Remark 11.1: the 3-step remediation ladder for persistent D_norm_y > 0.5,
    evaluated in STRICT CAUSAL ORDER (ε position → ρ seed/s → N_q/ESS/anneal T),
  * B3(iii): the α ≡ ρ identity-escape early-stop (Stdt(α_t) < 0.02 after the
    anneal has reached its floor for `patience` consecutive epochs),
  * The §13 Failure Catalogue (11 rows) as an ablation registry: each row is
    either a runnable config/adapter switch (default = the healthy setting),
    or UNREPRESENTABLE by construction with the unit test that pins it.

Nothing here mutates a run; the loop owners (train.py, the runbook driver)
call these on per-epoch diagnostics and act on the verdicts loudly.
"""

import math

# ---------------------------------------------------------------- Table 1 (§11.5)

PROBE_Z_BAND = (0.70, 0.90)      # linear subject probe on z_t (sanity: content exists)
PROBE_H_TARGET_MAX = 0.20        # probe on h'_t: < 15-20% (near 1/S chance)
PROBE_H_FAILURE = 0.45           # >45% => rank m too small or U stale (Table 1 row 1)
ZERO_STEP_ACC_MIN = 0.38         # nearest-centroid bag acc once the gate opens
ZERO_STEP_FLAT_MAX = 0.25        # flat at 20-25% => no emotion clustering in span(U)^perp
ENERGY_BAND = (0.35, 0.75)       # E||h'||^2 / E||z-mu||^2
ENERGY_SHRINK = 0.10             # <0.10 => nullspace shrinkage toward tau_norm
STDT_MIN = 0.15                  # intra-trial Stdt(alpha_t)
STDT_ESCAPE = 0.02               # <0.02 => alpha == rho_y (uninformative gate)
GATE_BAND = (0.4, 0.9)
DNORM_BAND = (0.1, 0.5)          # D_norm_y > 0.5 persistent triggers the ladder

# ---------------------------------------------------------------- Remark 11.2 smoke gate

SMOKE = dict(
    probe_z_min=0.80,            # subject probe on z_t must start > 80%
    probe_h_max=0.30,            # and drop < 30% on h'_t (chance = 1/S ~ 20%)
    stdt_min=0.15,               # intra-trial Stdt(alpha_t) > 0.15
    unseen_bag_acc_min=0.35,     # unseen-subject bag acc crosses 35-40%+
)

# ---------------------------------------------------------------- Remark 11.1 ladder

# epsilon "near eps_min" margin: within this fraction of the interval width
LADDER_EPS_NEAR_MIN = 0.25
# rho "under-seeded near rho_min" slack: rho_y <= rho_min * (1 + this)
LADDER_RHO_SLACK = 0.5
# ESS shortfall: ESS_y < frac * N_q * rho_tilde_min collapses onto stale outliers
LADDER_ESS_FRAC = 0.5


def evaluate_table1(record):
    """Grade one epoch's Table-1 record. `record` keys (floats unless noted):
    probe_z, probe_h, zero_step_acc, energy_ratio, stdt_alpha, gate_g,
    dnorm (dict class->D_norm_y), rho (dict or list class->rho_y),
    rho_order_pairs (optional list of (low_y, high_y) pairs that must satisfy
    rho[low] < rho[high], e.g. (Fear, Neutral), (Disgust, Neutral)).
    Returns {"status": "ok"|"fail", "issues": [str, ...]}. Missing keys RAISE —
    a diagnostic that was never computed must not grade as silent agreement.
    """
    for k in ("probe_z", "probe_h", "zero_step_acc", "energy_ratio",
              "stdt_alpha", "gate_g", "dnorm", "rho"):
        if k not in record:
            raise KeyError(f"evaluate_table1: missing diagnostic {k!r} — Table 1 "
                           "is graded only on metrics that were actually measured")
    issues = []
    if not (PROBE_Z_BAND[0] <= record["probe_z"] <= PROBE_Z_BAND[1]):
        issues.append(f"probe_z={record['probe_z']:.3f} outside {PROBE_Z_BAND} (no subject "
                      "content to erase, or pathology)")
    if record["probe_h"] > PROBE_H_FAILURE:
        issues.append(f"probe_h={record['probe_h']:.3f} > {PROBE_H_FAILURE}: rank m too "
                      "small or U stale (Table 1 row 1 failure mode)")
    elif record["probe_h"] > PROBE_H_TARGET_MAX:
        issues.append(f"probe_h={record['probe_h']:.3f} above the 15-20% target band")
    if record["zero_step_acc"] < ZERO_STEP_FLAT_MAX:
        issues.append(f"zero_step_acc={record['zero_step_acc']:.3f} flat at chance: no "
                      "emotion clustering in span(U)^perp")
    elif record["zero_step_acc"] < ZERO_STEP_ACC_MIN:
        issues.append(f"zero_step_acc={record['zero_step_acc']:.3f} below {ZERO_STEP_ACC_MIN}")
    if record["energy_ratio"] < ENERGY_SHRINK:
        issues.append(f"energy_ratio={record['energy_ratio']:.3f} < {ENERGY_SHRINK}: "
                      "nullspace shrinkage toward tau_norm")
    elif not (ENERGY_BAND[0] <= record["energy_ratio"] <= ENERGY_BAND[1]):
        issues.append(f"energy_ratio={record['energy_ratio']:.3f} outside {ENERGY_BAND}")
    if record["stdt_alpha"] < STDT_ESCAPE:
        issues.append(f"stdt_alpha={record['stdt_alpha']:.3f} < {STDT_ESCAPE}: alpha == rho "
                      "(uninformative gate — B3(iii) escape)")
    elif record["stdt_alpha"] < STDT_MIN:
        issues.append(f"stdt_alpha={record['stdt_alpha']:.3f} below {STDT_MIN}")
    if not (GATE_BAND[0] <= record["gate_g"] <= GATE_BAND[1]):
        issues.append(f"gate_g={record['gate_g']:.3f} outside {GATE_BAND}")
    rho = record["rho"]
    for y, d in record["dnorm"].items():
        if d > DNORM_BAND[1]:
            r_y = rho[y] if not isinstance(rho, dict) else rho[y]
            issues.append(f"D_norm[{y}]={d:.3f} > {DNORM_BAND[1]} persistent -> run "
                          f"remediation_ladder(dnorm_y={d:.3f}, eps..., rho_y={r_y}, ess_y=...)")
        elif d < DNORM_BAND[0]:
            issues.append(f"D_norm[{y}]={d:.3f} < {DNORM_BAND[0]}: drop channel idle")
    for low, high in record.get("rho_order_pairs", ()):
        if not (rho[low] < rho[high]):
            issues.append(f"rho[{low}]={rho[low]:.3f} !< rho[{high}]={rho[high]:.3f} — "
                          "sparsity term not separating Fear/Disgust from Neutral")
    return {"status": "ok" if not issues else "fail", "issues": issues}


def evaluate_smoke(record):
    """Remark 11.2 pass criteria; record keys: probe_z, probe_h, stdt_alpha,
    unseen_bag_acc. Returns (passed, report) with a per-criterion verdict."""
    for k in ("probe_z", "probe_h", "stdt_alpha", "unseen_bag_acc"):
        if k not in record:
            raise KeyError(f"evaluate_smoke: missing criterion input {k!r}")
    report = {
        "probe_z_kept":   record["probe_z"] >= SMOKE["probe_z_min"],
        "probe_h_dropped": record["probe_h"] < SMOKE["probe_h_max"],
        "stdt_informative": record["stdt_alpha"] > SMOKE["stdt_min"],
        "unseen_bag_acc": record["unseen_bag_acc"] >= SMOKE["unseen_bag_acc_min"],
    }
    return all(report.values()), report


def remediation_ladder(dnorm_y, eps, eps_min, eps_max, rho_y, rho_min,
                       ess_y, n_q, rho_tilde_min):
    """Remark 11.1 in STRICT causal order. Caller must pass the class y whose
    D_norm_y > 0.5 persists after warmup; all quantities are the ACTUAL run's
    (Thm 12.2 interval endpoints, Option-B rho, queue ESS, sizing floor).
    Returns {"steps": [ {step, fires, action, evidence}, ... ], "next_step",
    "next_action"} — the first firing step in causal order wins; when steps 1-3
    are all healthy yet D_norm > 0.5, step 4 (escalate) fires: the ladder
    refuses a silent "all clear" on a persistent failure."""
    if not (dnorm_y > DNORM_BAND[1]):
        raise ValueError(f"remediation_ladder called with D_norm_y={dnorm_y:.3f} — the "
                         "ladder triggers only on persistent D_norm_y > 0.5 (Table 1 row 5). "
                         "Do not run remediation on a healthy class.")
    if eps_max < eps_min:
        raise ValueError(f"eps_max ({eps_max}) < eps_min ({eps_min}) — Thm 12.2 interval corrupt")
    width = max(eps_max - eps_min, 1e-12)
    near_min = eps <= eps_min + LADDER_EPS_NEAR_MIN * width
    under_seeded = rho_y <= rho_min * (1.0 + LADDER_RHO_SLACK)
    ess_floor = LADDER_ESS_FRAC * n_q * rho_tilde_min
    steps = [
        {"step": 1, "fires": bool(near_min),
         "action": f"epsilon={eps:.4f} sits near eps_min={eps_min:.4f} (interval "
                   f"[{eps_min:.4f}, {eps_max:.4f}]): early cosine costs give "
                   "exp(-C/eps) << 1-rho_y, forcing mass into the drop slack d_i. "
                   "SHIFT epsilon toward the midpoint of its admissible interval.",
         "evidence": f"(eps-eps_min)/width = {(eps - eps_min) / width:.3f} <= {LADDER_EPS_NEAR_MIN}"},
        {"step": 2, "fires": bool(under_seeded),
         "action": f"rho_y={rho_y:.4f} under-seeded near rho_min={rho_min:.4f} => "
                   "tau_y = -eps*log(1-rho_y) too small, dropping nearly free. "
                   "RE-VERIFY the Option-B fixed-point seed and the burst spread s > 1.",
         "evidence": f"rho_y <= rho_min*(1+{LADDER_RHO_SLACK})"},
        {"step": 3, "fires": bool(ess_y < ess_floor),
         "action": f"ESS_y={ess_y:.1f} << prototype-richness floor {ess_floor:.1f} "
                   f"(= {LADDER_ESS_FRAC}*N_q*rho_tilde_min, N_q={n_q}): nu_y collapsed "
                   "onto a few stale high-beta outliers. INCREASE N_q toward the ceiling "
                   "or RAISE the E-step annealing temperature T.",
         "evidence": f"ESS_y = (sum beta)^2/sum beta^2 vs floor {ess_floor:.1f}"},
        {"step": 4, "fires": not (near_min or under_seeded or ess_y < ess_floor),
         "action": "Ladder exhausted (eps well-positioned, rho seeded, ESS rich) yet "
                   "D_norm_y persists > 0.5: escalate to the full §11.5 diagnostic "
                   "review — this is NOT a 'carry on' verdict.",
         "evidence": f"dnorm={dnorm_y:.3f}, steps 1-3 all healthy"},
    ]
    first = next((s for s in steps if s["fires"]), None)
    return {"triggered_class_dnorm": float(dnorm_y), "steps": steps,
            "next_action": first["action"] if first else None,
            "next_step": first["step"] if first else None}


def alpha_identity_escape(stdt_series, anneal_done_flags, patience=3):
    """B3(iii) early-stop: if the anneal has reached its floor AND
    Stdt(alpha) < 0.02 for `patience` CONSECUTIVE epochs, alpha has degenerated
    to the constant rho prior — the gate carries no within-trial information
    and every downstream term runs on an uncalibrated posterior. Returns the
    stop index or None."""
    if len(stdt_series) != len(anneal_done_flags):
        raise ValueError("alpha_identity_escape: series/flag length mismatch")
    streak = 0
    for i, (s, done) in enumerate(zip(stdt_series, anneal_done_flags)):
        if done and s < STDT_ESCAPE:
            streak += 1
            if streak >= patience:
                return i
        else:
            streak = 0
    return None


# ---------------------------------------------------------------- §13 ablation registry

FAILURE_CATALOGUE_ROWS = [
    {"row": "Remove L_anc (Projected DCL)",
     "how": "config: loss.sm.lambda_anc = 0.0",
     "expect": "constant-vector / nullspace collapse (cMMD dead config)"},
    {"row": "Remove unbalanced dropping (d_i, u_j)",
     "how": "config: loss.sm.force_balanced = true (drop price pinned => dustbin inert "
            "=> balanced OT)",
     "expect": "SCDA subject mining returns (~87% probe)"},
    {"row": "Remove nullspace projection (L_sub, I-UU^T)",
     "how": "config: loss.sm.warmup_epochs >= total epochs (the projector stays the "
            "identity for the whole run and L_sub is rank-forced to 0 per §11.4ii; add "
            "loss.sm.lambda_sub = 0.0 to also kill the residual penalty)",
     "expect": "subject memorisation; probe -> 91%"},
    {"row": "Split 1A queue (128-D) vs 1B readout (256-D)",
     "how": "UNREPRESENTABLE as a dimension split (every consumer goes through "
            "NullspaceState.project); the sibling projected-vs-raw split is config: "
            "loss.sm.anchor_on = raw",
     "expect": "backbone stashes subject ID in the MLP-head nullspace"},
    {"row": "Evaluate log p_s on full space instead of h'_t",
     "how": "UNREPRESENTABLE: log_baseline_gaussian takes nullspace coordinates only",
     "expect": "within-trial jitter in span(U) falsely triggers the burst gate alpha_t",
     "pinned_by": "tests/test_stage3.py (baseline is nullspace-only by signature)"},
    {"row": "Remove sparsity (L_spr)",
     "how": "config: loss.sm.lambda_rho = 0.0",
     "expect": "rho_y unconstrained; gaslighting on Fear/Disgust"},
    {"row": "Replace exact E-step with learned alpha-head",
     "how": "PATH REMOVED from the repo (SubjectEraserLoss/WindowReRanker deleted, Stage 0)",
     "expect": "calibration failure; OT activates on an uncalibrated gate",
     "pinned_by": "audit §4/§5; retired flags raise at config load"},
    {"row": "Omit 1/T_b normalisation in R_b",
     "how": "adapter flag: Phase1BReadout(..., omit_duration_norm=True) [ablation only]",
     "expect": "trial-duration bias: ||R|| grows with T_b",
     "pinned_by": "tests/test_stage7.py duplication-invariance vs >3x inflation guard"},
    {"row": "Omit b_y in the 1B head",
     "how": "adapter flag: init_bayes_head(ablate_bias=True) [ablation only]",
     "expect": "burst-rate magnitude rho_y lost; largest-rate class collapse",
     "pinned_by": "tests/test_stage7.py burst-rate bias row"},
    {"row": "Back-prop through the Sinkhorn fixed point",
     "how": "UNREPRESENTABLE: class_ot_term solves under no_grad; grads re-enter only "
            "through the stop-grad-coupled costs (Prop 8.3)",
     "expect": "NaN blowup through the fixed point (DANN/GRL dead config)",
     "pinned_by": "tests/test_stage4.py no_grad guard + coupling gradient tests"},
    {"row": "Queue-vs-queue OT",
     "how": "UNREPRESENTABLE: class_ot_term raises when torch.is_grad_enabled() is False "
            "for the query side / queue-only pairs are never constructed",
     "expect": "zero gradient; dead OT term",
     "pinned_by": "tests/test_stage4.py (Prop 9.1 guard)"},
]


def ablation_switch(name, overrides):
    """Return the config overlay for a catalogue row after CHECKING it exists in
    the registry — a typo'd ablation must not silently run as the healthy config."""
    hit = [r for r in FAILURE_CATALOGUE_ROWS if r["row"] == name]
    if not hit:
        raise KeyError(f"ablation {name!r} not in FAILURE_CATALOGUE_ROWS "
                       f"({len(FAILURE_CATALOGUE_ROWS)} rows)")
    bad = [k for k in overrides if not (k.startswith("loss.sm") or k.startswith("model") or k.startswith("training"))]
    if bad:
        raise ValueError(f"ablation {name!r}: non-loss overrides {bad} need runbook review, not the registry")
    return {"row": name, "expect": hit[0]["expect"], "overrides": dict(overrides)}
