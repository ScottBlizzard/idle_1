# Publication scope and cleanup — 2026-09-07

This is a review snapshot built from implementation commit
`c941f40` plus 50 explicitly inventoried local implementation/engineering files,
the current manuscript sources and supporting notes, the new 20-day roadmap,
and the numerical repair handoff/test receipt. The original local working trees
and server runs were not modified by publication preparation.

The GitHub default branch previously pointed to `3bdeac0`. Its history is retained.
Publication uses an ordinary fast-forward push; no history rewrite is intended.

## Cleaned from the current tree

- Three superseded August 5 prompt documents moved to `archive/prompts/`.
- The root review entry point now describes the September 7 strategy review.
- Eighteen old console logs removed from the publication tree. They remain in
  commit `c941f40` and the original local worktree; associated scientific summaries,
  decisions and structured result artifacts are retained.
- Additional ignore patterns exclude worktrees, bundles, tool caches, environments,
  native binaries and manuscript build products from accidental future staging.
- Untracked local model tensors, large run directories, PDF build outputs, temporary
  archives and unrelated root experiments were not swept into the publication.

Removed console logs (recover via `git show c941f40:<path>`):
- `analysis/GREEN_V134_TERMINAL_ARCHIVE_20260825/development/worker_00/worker.log`
- `analysis/GREEN_V134_TERMINAL_ARCHIVE_20260825/development/worker_01/worker.log`
- `analysis/GREEN_V134_TERMINAL_ARCHIVE_20260825/development/worker_02/worker.log`
- `analysis/GREEN_V134_TERMINAL_ARCHIVE_20260825/development/worker_03/worker.log`
- `analysis/GREEN_V134_TERMINAL_ARCHIVE_20260825/development/worker_04/worker.log`
- `analysis/GREEN_V134_TERMINAL_ARCHIVE_20260825/development/worker_05/worker.log`
- `analysis/GREEN_V134_TERMINAL_ARCHIVE_20260825/development/worker_06/worker.log`
- `analysis/GREEN_V134_TERMINAL_ARCHIVE_20260825/development/worker_07/worker.log`
- `analysis/GREEN_V135_TERMINAL_ARCHIVE_20260825/development/worker_00/worker.log`
- `analysis/GREEN_V135_TERMINAL_ARCHIVE_20260825/development/worker_01/worker.log`
- `analysis/GREEN_V135_TERMINAL_ARCHIVE_20260825/development/worker_02/worker.log`
- `analysis/GREEN_V135_TERMINAL_ARCHIVE_20260825/development/worker_03/worker.log`
- `analysis/GREEN_V135_TERMINAL_ARCHIVE_20260825/development/worker_04/worker.log`
- `analysis/GREEN_V135_TERMINAL_ARCHIVE_20260825/development/worker_05/worker.log`
- `analysis/GREEN_V135_TERMINAL_ARCHIVE_20260825/development/worker_06/worker.log`
- `analysis/GREEN_V135_TERMINAL_ARCHIVE_20260825/development/worker_07/worker.log`
- `analysis/GREEN_V300_FORMAL_PREPARE_20260826/green_v300_combined_272_tests.log`
- `analysis/archive/green_v13_stop_20260825/green_bridge_v13_prepare.log`

## Evidence intentionally retained

July audit; IRS controls; failed v1/v2/v3 decisions; corrected v3 machine-readable
results; development completeness failure; strong baseline findings; original
mission and novelty audit; current repaired implementation and tests. Negative
findings are relevant scientific evidence, not clutter.

Historical files can contain obsolete paths, instructions, thresholds or dates.
They are preserved as records of their stages, not current operating instructions.
Large historical folders are outside the main reading path, but retain their
locations so evidence references and machine-readable lineage remain usable.

## Checks and limitations

The current entry points identify source evidence and known gaps explicitly.
The 92-test XML is an existing execution receipt, not a newly run full-project test.
Repository cleanup itself does not validate the scientific claims. The reviewer
must inspect source and independently check important bibliography/novelty claims.

No confirmation outcomes were added, no paper results invented, and no active
server job restarted. Native library binaries are intentionally omitted; build
instructions and source remain in the repository.
