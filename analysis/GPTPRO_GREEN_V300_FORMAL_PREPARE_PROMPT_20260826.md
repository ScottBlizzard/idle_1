# GPTPro GREEN v3.0.0 Formal Prepare Review Prompt

Use the following prompt verbatim.

```text
You are the binding scientific and protocol reviewer for the GREEN project.

Repository:
https://github.com/ScottBlizzard/idle_1

Branch:
codex/green-v300

Frozen formal-evidence archive commit:
7abbbcb39465cb83e1cc7452ede080c2f4862b78

Implementation commit used by the formal prepare:
602e7c9c5e43f1aecf1485d4ad2fc6574b2fdaa1

Please inspect the repository and all relevant source and evidence directly.
At minimum, read:

1. analysis/GPTPRO_GREEN_V21_POSTMORTEM_DECISION_20260825.md
2. analysis/GREEN_V21_POSTMORTEM_20260825/POSTMORTEM_SUMMARY.md
3. analysis/CODEX_GREEN_V300_CANONICAL_PAYLOAD_CORRIGENDUM_20260826.md
4. analysis/CODEX_GREEN_V300_FLOAT64_RESPONSE_CORRIGENDUM_20260826.md
5. analysis/GREEN_V300_FORMAL_PREPARE_20260826/FORMAL_PREPARE_SUMMARY.md
6. every file in analysis/GREEN_V300_FORMAL_PREPARE_20260826/, including
   the NPZ direction archive, formal manifest, sha256sums, radius rows,
   theorem rows, Gate-04 audit, hardware/throughput evidence, and exact
   272-test log
7. the complete GREEN v3 implementation under src/, especially
   green_bridge_v300_prepare.py, exp_green_bridge_v300.py,
   green_bridge_v300_spec.py, green_bridge_v300_transport.py,
   green_bridge_v300_numerics.py, green_bridge_v300_directions.py,
   analyze_green_bridge_v300.py, green_bridge_v300_multigpu_worker.py,
   and launch_green_bridge_v300.sh
8. tests/test_green_bridge_v300_contract.py

Verified formal status:

- The unique formal prepare ran at implementation commit 602e7c9... on
  physical RTX 4090 GPU 4.
- prepare_result.json reports PREPARE_PASS.
- formal_one_shot=true, attempt_index=1, retry_allowed=false.
- development_started=false and confirmation_started=false.
- The exact combined suite passed 272/272 tests with zero skips.
- All 21 non-self-referential formal artifact hashes pass locally and on
  the server.
- The formal root contains all 22 required protocol artifacts.
- The selected outcome-blind global radius is rho*=1.
- At rho=1, 360/360 object-stratum fidelity checks pass.
- The maximum finite-versus-AD difference divided by its eligibility
  ceiling at rho=1 is 0.002004559655602945.
- Exact direct-transport AD routes: zero failures.
- Exact direct-transport theorem checks: zero failures.
- Exact all-ten joint-composition routes: zero failures.
- Exact all-ten joint-composition checks: zero failures.
- Peak allocated GPU memory is 7.696776390075684 GiB.
- Projected prepare + development + confirmation time is
  2023.380114857573 seconds.
- No v3 development or confirmation response, anchor, derivative, cache,
  or timing value has been generated.

Important numerical correction requiring explicit scientific review:

The initial complete legacy-only float32 dry run had zero AD route failures
and zero exact theorem failures, but no eligible global radius. Failure was
localized to C and delta_H second-order finite stencils and worsened as the
radius decreased. G, J_path, and J_control remained accurate. This is the
signature of catastrophic cancellation of quantized float32 endpoint values,
not a matched-bypass theorem failure.

Codex therefore made an implementation-level precision upgrade without
changing any scientific threshold, mathematical object, candidate radius,
split, record, gate, held-out direction, success rule, baseline, or phase
authorization:

- the response-only finite point estimator evaluates preregistered stencil
  endpoints on an isolated float64 copy of the frozen model tail;
- the active float32 model remains byte-identical;
- a second, distinct isolated float64 copy provides dual-route AD targets;
- the finite point estimator calls no jacfwd, jacrev, JVP, VJP, automatic
  derivative, white-box gradient, or AD value;
- AD remains audit/target only and never becomes the point estimator.

This correction made rho=1, 1/2, 1/4, and 1/8 fully eligible, with all
360/360 checks passing at each radius. The frozen largest-eligible rule then
selected rho*=1. Full provenance is included in the two Codex corrigenda and
formal artifacts.

Please issue a self-contained, binding scientific decision that does all of
the following:

A. Independently audit whether the canonical-payload correction is a valid
   non-scientific reproducibility correction.

B. Independently audit whether the isolated float64 response-only estimator
   is scientifically admissible as a numerical precision upgrade, rather
   than AD leakage or a post hoc threshold change. Explicitly assess likely
   ICLR reviewer objections and the strongest defensible wording.

C. Verify or reject the formal PREPARE_PASS from the committed artifacts.
   Check hashes, split/firewall status, direction design, rho selection,
   theorem preflights, memory/runtime, test count, and absence of inferential
   artifacts.

D. If the corrections and formal prepare are valid, decide whether GREEN
   v3.0.0 development may now be authorized exactly once. If authorized,
   provide a complete implementation-ready DEVELOPMENT-ONLY protocol:

   - exact authorized commit/ancestry rules;
   - exact use of rho*=1 and finite-response precision;
   - exact worker/GPU allocation;
   - exact input/output schemas, filenames, manifests, and hashes;
   - exact endpoint ledger and crash semantics;
   - exact aggregation and frozen-baseline rules;
   - exact gate ordering and first-failure behavior;
   - exact commands;
   - exact return bundle;
   - exact STOP conditions.

E. Do not authorize confirmation in this decision. Confirmation must remain
   sealed even if development passes.

F. If the numerical correction or formal prepare is not acceptable, do not
   rewrite or retry the consumed formal v3.0.0 prepare. State whether a new
   protocol identity is required and provide exact correction requirements.

Preserve the paper's Oral-level theoretical ambition. Do not downgrade the
main theorem, causal-transport estimand, or held-out transport story. Do not
relax a threshold merely to obtain a pass. Clearly distinguish binding
requirements, recommendations, forbidden actions, and authorization state.

Write the complete final decision into one Markdown document named:

analysis/GPTPRO_GREEN_V300_FORMAL_PREPARE_DECISION_20260826.md

Do not return only a chat summary. The Markdown file must contain the entire
binding decision and be executable without unstated interpretation.
```
