# GREEN Bridge protocol v1.2 — server STOP report

Date: 2026-08-05  
Server execution commit: `c40405122c779337f44b811c42850b36ba5ff850`  
Reviewed ancestor: `b87300a6f56cb4706db090486d8bec77a2fc2b23`  
Binding decision: `analysis/GPTPRO_GREEN_GATE08_DECISION_20260805.md`  
Branch: `main`

## Terminal verdict

The unique ledger-backed rank-five run stopped at the first new scientific
basis gate:

```json
{
  "schema_version": "green-bridge-terminal-v1",
  "verdict": "STOP",
  "first_failed_gate": "08B_BASIS_FIT_SPECTRUM",
  "detail": "sigma5/sigma6=1.0227285601080833, sigma5/sigma1=0.535667052214108",
  "time_utc": "2026-08-05T14:29:33Z"
}
```

Thus the fit spectrum had a substantial fifth direction but no preregistered
five-versus-six separation:

- `sigma5/sigma6 = 1.0227285601080833 < 1.10`;
- `sigma5/sigma1 = 0.535667052214108 >= 1e-4`.

No retry, donor replacement, threshold change, or rank-six fallback was made.

## One-run and provenance evidence

The run ledger was created before model responses and records:

```json
{
  "protocol_run_id": "green-bridge-v1.2-one-shot",
  "attempt_index": 1,
  "retry_allowed": false,
  "prepare_restart_allowed": false,
  "development_restart_allowed": false,
  "confirmation_restart_allowed": false,
  "prepare_started": true,
  "development_started": false,
  "confirmation_started": false,
  "execution_commit": "c40405122c779337f44b811c42850b36ba5ff850",
  "created_utc": "2026-08-05T14:27:56Z"
}
```

The initial manifest records `repository_dirty_at_launch=false`, branch
`main`, empty status porcelain, and the required review commit as an ancestor.
The server worktree remained clean after termination. GPU 4 returned to its
225 MiB idle baseline.

There was one non-scientific launcher preflight rejection immediately before
the ledger-backed run: the wrapper itself wrote
`logs/green_bridge_v12_prepare.pid` inside the worktree, so Gate
`00_REPOSITORY_CLEAN` rejected it. This happened before the output root, run
ledger, tokenizer, model, or any response existed. The wrapper-generated PID
file was removed and the PID was moved to `/tmp`; the run above is the only
run that crossed the clean gate and created a ledger.

## Historical Gate-04 replay

The frozen legacy panel replay passed unchanged:

- audit version: `hf-tl-fidelity-v2`;
- eager attention, batch size 1, 32 prompts;
- ordered prompt hash:
  `619d21c10d4f30e6ce2597c3ba4df1de72cf0cb4f6cce322d82c2d3ec62803ce`;
- exact weight mapping passed;
- same-TransformerLens no-op maximum absolute error: `0.0`;
- HF–TL error remained excluded from `epsilon_y`.

Observed Gate-04 metrics:

| Metric | max abs | pooled RMS / RMS |
|---|---:|---:|
| raw year logits | 1.068115234375e-4 | 3.124910032944241e-5 |
| centered year logits | 4.9591064453125e-5 | 1.1493354309025036e-5 |
| task margin | 9.78552776834239e-6 | 3.832678665827661e-6 |
| resid_mid | 6.4849853515625e-5 | 3.66433163993219e-6 |
| selected pre | 3.337860107421875e-6 | 1.0100865974837965e-6 |
| selected post | 3.337860107421875e-6 | 8.878869914757482e-7 |

## Donor-v2 plan evidence

The tokenizer-filtered plan passed all preregistered construction checks:

- counts: 512 `basis_fit`, 256 `basis_holdout`, 512 `radius_v2`;
- 1,280 total pairs and 2,560 unique prompts;
- zero prompt overlap;
- zero overlap with evaluation nouns;
- zero overlap with legacy donor nouns;
- full plan SHA-256:
  `8819fb32e516e982c14ae2877c0dcb680237bdf16c12506bf3f871eb9a7f24cb`;
- fit ordered keys:
  `ee1e7cf9c5cc185df5acffd78b7432068a061cb5a4883d39e92b82acf52eeac5`;
- holdout ordered keys:
  `42ca7839e584335f1a0601a1430bac40a9497d39d798d1d038c7714a381f1136`;
- radius ordered keys:
  `32e697f9783bbbcbca908065422dca343a2063ddb9226966a6f4a040ce510b4e`;
- all prompt keys:
  `ccc988189caa49ef90fffb447eebdc7ff13fc83742f9b46784c8fb55d4ee39c5`.

The unchanged tokenizer-filtered evaluation plan SHA-256 is
`150f146ef69858bce77677ce74a4806129720ee68395246cbce91d498f06960c`.
The reconstructed legacy donor plan SHA-256 is
`a39df1cdb4dbb36c1cb2b8c98c58b729bf5f358793b2e4bcadf00fc24cc2ed95`.

## Data-access boundary

The following did not exist at termination:

```text
noise_audit_dev.json
dev_tensor_scores.parquet
dev_energy_targets.parquet
dev_result.json
frozen_analysis.json
confirm_tensor_scores.parquet
confirm_energy_targets.parquet
```

`confirmation_open=false`, `prepare_complete` was never set, and neither
development nor confirmation was claimed in the ledger. No development or
confirmation response was observed.

## Artifact hashes

```text
390c5b62d5b42e216abbb15a0d6d206a55419c48117f610f34c0ac802e153747  result.json
ea486fe8eea798b16951fcea9394b1c4ddb4b44bd4afb5c8b104b37aaf047be  manifest.json
49aa7a1818fb06d63b975938aea7285d3198fccc97723a96a37afa097abdbb99  hook_audit.json
2c8dd401b93d3864969ab941b85cae2ab5e6e983bdf39b909f33c532b480cc16  donor_v2_plan.json
fa88911fcce749942a24c9e479c66cf89cd72ce9386b76d146262de6671b4f65  run_ledger.json
84d78b0ffefb0761138c447f19d594cb364304563b73f068ee85b58bb5c1b9ec  splits.json
7fb05a1bf83d0083c622630694df09485dbaf18f4caaf6f5614200e0d8d2baf0  development_splits.json
646d2ebcf1229645c83ebadea7f39d782e12152a8248dbd122f8c11e58c83df1  gate04_legacy_panel.json
fb9bd5a686d1bb09fa31c4cc308ff51f26c1d64075feb57d5a330db8fcaa6cb0  model_fingerprint.json
dfa25ab6bdc067a223973379cd0f1fb038426e8f0f3b26c1dd77dda264b69805  first_order_directions.npy
07eca7773602d1fceea18e7080be4e6ce83c61f55caf4882c01b071d9232b18d  green_bridge_v12_prepare.log
```

## Verification completed before execution

- local binding CPU suite: 174/174 passed;
- local `pytest -q src`: 207/207 passed;
- server binding CPU suite: 174/174 passed;
- Python 3.11.13, Torch 2.7.1+cu126, CUDA 12.6;
- TransformerLens 3.6.0, Transformers 5.13.0, NumPy 2.2.6,
  SciPy 1.15.3, pandas 2.2.3, pyarrow 19.0.1, threadpoolctl 3.6.0.

## Required next action

The rank-five oral line is terminated under the binding one-shot rule. GPTPro
must decide the next scientific action from this donor-only failure. Local or
server execution must not resume from this output root, and no new basis run
may be started without a new explicit decision.

## Post-stop conformance observation

One implementation-level serialization deviation was identified during final
report review. `build_basis_and_radii()` constructs the float64 fit, holdout,
and radius matrices before calling `fit_rank5_basis()`, but computes and writes
`donor_v2_matrix_hashes.json` only after that call returns. Because the fit
spectrum gate raised inside `fit_rank5_basis()`, the stopped run did not
serialize the three response-matrix hashes, shapes, or the full singular-value
array. The two reported ratios were computed directly by the frozen SciPy
`gesvd` path, and the ordered donor plan and all prompt keys are hashed, so this
ordering defect has no numerical route to change `1.0227285601080833`.
Nevertheless, it violates the decision's requirement to hash response
matrices before SVD and weakens post-stop auditability. It does not authorize a
retry; GPTPro must account for it in the next binding decision.
