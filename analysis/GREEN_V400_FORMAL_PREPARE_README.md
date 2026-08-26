# GREEN v4 static formal prepare

This branch implements only the theorem and outcome-blind preparation package
authorized by `GPTPRO_GREEN_V400_BINDING_CORRIGENDUM_20260826.md`, interpreted
through `CODEX_GREEN_V400_EXECUTION_CONSISTENCY_LOCK_20260826.md`.

The sole launch surface is:

```bash
bash scripts/launch_green_bridge_v400_formal_prepare.sh
```

The launcher requires a clean
`codex/green-v400-joint-witness-formal-prepare` branch descending from
`48182844a43d391439704f27aa26d513d33adaa0`. All environments, caches, temporary
files, logs, and outputs are resolved below `/mnt/sdb`. The immutable formal
package is written to:

```text
/mnt/sdb/ccj/outputs/green_bridge_v400_formal_prepare/
```

The server's existing frozen-model Python is reused read-only. Pinned
validated-numerics additions are installed with `--target` below `/mnt/sdb/ccj`
because the system Python has no `ensurepip`; this is recorded in the formal
engineering-correction ledger.

The run performs the inherited regression, the exact 70-test theorem barrier,
model/token/hook static preflights, sealed-set exclusion, row-universe freezing,
static causal-cone planning, and an independent read-only audit. It may hash
t-independent donor hook geometry. It does not run a real-row certificate,
select `q`, inspect a development response, execute P13, or access confirmation
content.

The only successful terminal state is
`PREPARE_PASS_STATIC_THEOREM_ONLY`, followed by the printed stop token
`STOP_AFTER_FORMAL_PREPARE_RETURN_TO_GPTPRO`. Development and confirmation need
separate later scientific authorization.
