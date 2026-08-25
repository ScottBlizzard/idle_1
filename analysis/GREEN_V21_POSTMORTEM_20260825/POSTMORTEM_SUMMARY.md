# GREEN v2.1 read-only postmortem summary

Status: `POSTMORTEM_PASS`  
Official v2.0.0 verdict: `STOP_ORAL` (unchanged)  
Confirmation data accessed: no  
Usable for v3 threshold selection: no

## Decisive proof checks

- The immutable v2 archive and full formal root passed every recorded SHA-256 check.
- Analysis 04 contains 19,200 gate-system-item-direction rows: 64 items, two systems, ten gates, five frozen-frame directions and ten deterministic held-out directions. Forward/reverse AD routes and the exact direct-transport theorem have zero failures. The maximum residual-to-bound ratio is `4.114108706298039e-05`.
- Analysis 05 contains 192 rows: `tar`, `pat`, and item-level `pat - tar` for every item. Exact all-ten joint composition has zero failures. The maximum residual-to-bound ratio is `2.3518002819813065e-09`.
- The active model fingerprint was unchanged in all eight GPU shards.

These checks reject a falsified matched-bypass transport identity or an AD implementation defect as the explanation for v2 failure.

## Failure localization

At item-level `pat - tar`, the exact AD response with the white-box gradient matches the independent joint AD target to floating-point precision (median relative error `1.6314703619662762e-15`). Replacing exact `G` by the fine finite-radius estimate while retaining the white-box gradient remains accurate (median relative error `0.0014929355273281468`). In contrast, the complete fine response-inverse estimator has median relative error `1.9579782379926258`.

At cell level, the exact AD response versus the independent finite-energy target has median relative error `0.2536195062958334`. This remaining gap is a target/regime transport issue, not an exact local theorem failure.

Therefore the primary diagnosis is finite-radius response identifiability, especially curvature inversion of `g`, with a secondary finite-energy target/regime mismatch. This supports the v3 curvature-controlled identifiability and held-out causal-transport design.

## Secondary audits

- The official v2 gate labels reconstruct exactly: 7 active, 1,262 certified target-null, and 11 unresolved.
- All 1,280 gate-system cases have finite fine and coarse inverses in the post-hoc ladder; these results are diagnostic only and may not select v3 thresholds.
- The noun-century cluster bootstrap interval for the overall same-role minus disjoint-role shift is `[-0.0003013367275851227, 0.0001777613461238908]`, which includes zero. Role sampling is not the primary explanation.
- v1.3.6 and v2 have similar median curvature, response, and residual-radius scales, but radically different label regimes. This supports a certification/identifiability-regime explanation rather than disappearance of the underlying response signal.
- The set-SNR value of one is mechanically reproduced from zero-crossing signed intervals.
- Prepare-panel, throughput, and full-development operation counts and timings all reconcile.

## Binding consequence

The v2 STOP remains immutable. The postmortem authorizes continued clean implementation and testing of GREEN v3.0.0 and, only after all prepare firewalls pass, one formal prepare run. It does not authorize v3 development or confirmation.
