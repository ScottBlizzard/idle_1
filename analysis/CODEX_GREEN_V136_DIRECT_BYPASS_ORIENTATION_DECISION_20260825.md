# GREEN v1.3.6 direct-bypass orientation decision

The unique v1.3.5 development attempt passed the v1.3.4 response-pairing site,
completed every energy record, and then terminated uniformly on the first
tensor record of each worker with `direct bypass and frame dimensions
disagree`. The identity remains closed.

The inverse stores `GateIdentification.D` in probe-by-output orientation
`[5,100]`, matching `J` and `P`. The frozen common-frame formula is written in
output-by-probe orientation:

```text
D_common = D_Q @ Q_gate.T @ Q_common
```

Therefore the executable helper must receive `identification.D.T`, with shape
`[100,5]`. v1.3.6 adds this transpose at the sole active call site. It does not
change any scientific input, estimator, threshold, intervention, split,
randomness, batching rule, or stopping rule.

The contract suite additionally drives the complete active-identified branch,
including response/cotangent pairing, direct-bypass common-frame mapping, and
error-bound accumulation, so these adjacent interfaces are checked together.
