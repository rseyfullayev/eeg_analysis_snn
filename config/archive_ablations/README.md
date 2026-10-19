# ARCHIVED ABLATIONS — read-only evidence

These configs reproduce **known-dead configurations** documented in `fixed.pdf` §1.2/§13
and the numbers cited in `PROJECT_STATE.md` / `milestones/`:

- `scda_mmd_muon.yaml` — cMMD → constant-vector collapse
- `scda_dann_muon.yaml` — DANN/GRL → NaN blowup (7×10¹⁵ grad norm)
- `scda_*.yaml` — SCDA subject-mining → 87% subject probe
- `scda_cosface_muon.yaml`, `supcon_partialbn_adamw.yaml` — superseded 1A variants

They remain Hydra-loadable (the 2026-10-19 B24 group-key renames — `train_size`,
`subject_diverse_k` — repaired the shared `data/` and `training/` groups they inherit),
but the model code they instantiate has since drifted (projector heads retired in the
SMC rebuild), so they are **evidence, not runnable experiment definitions**. Do not use
them for new runs; do not delete them — they are the paper's dead-list provenance.

See `REVIEW_SMC_AUDIT.md` §2.3 and §7 item 4.
