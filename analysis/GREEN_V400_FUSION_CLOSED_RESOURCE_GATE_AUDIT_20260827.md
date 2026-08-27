# GREEN v4.0.0 fusion-closed resource-gate audit — 2026-08-27

## Decision

- **GO:** the outcome-blind GPT-2 joint-witness TensorProgram is closed at 81 nodes, exact zero control, fixed softmax pivot 0, final unpadded causal position, element dependency masks, tensor-store references, semantic dispatcher signature, and exact final-contrast fusion.
- **GO:** the frozen 100,000,000 MPFR-operation cap is infeasible. The dense coefficient-arithmetic lower bound is 341,397,504 operations per precision/cell at sequence length 12; two mandatory initial cells already require at least 682,795,008 at one precision.
- **GO (narrow scope):** the native synthetic resident proxy executes the same ordered kernel-tag vector as the 81-node program and counts exactly 352,275,450 directed-enclosure arithmetic primitives per precision/cell under the frozen taxonomy.
- **GO (correctness scope):** the actual Python MPFR TensorProgram dispatcher emits all 81 semantic events only after successful node completion at both 384 and 512 bits. The event vector matches the program dispatcher signature and all five compiled/reference roots remain bit-identical on the deterministic tiny fixture.
- **GO (prepare-only scope):** the actual GPT-2 constants needed by the 81-node program are closed into one immutable 64-byte-aligned packed blob: 32 resident tensor records plus four exact-fusion source tensors cover all 36 unique program tensor references and all 150 node/input bindings. The packed plan is validated but remains explicitly `native_execution_ready=false`.
- **GO (narrow resident-fusion correctness scope):** the compiled dispatcher consumes the plan's exact fused final contrast without loading the four full-unembedding source tensors; all five roots remain bit-identical to the reference dispatcher at 384 and 512 bits on the deterministic tiny fixture.
- **GO (resident Python-dispatch correctness scope):** after plan validation, all 32 packed constants are consumed through aligned read-only mmap arrays. Each precision records 134 packed tensor bindings, zero tensor-store fallback reads, and four fused-contrast nodes. Every pairwise-affine node uses the packed batch ABI; exact backward row liveness materializes 148 of 228 dense axis-0 row slots; per-cell static-row caching reuses four layer-normalization rows and eleven affine rows. Both precisions preserve all five exact roots and the complete 81-event successful-dispatch trace.
- **GO (observation-only):** the native packed affine ABI was exercised with every distinct actual affine weight/bias pair used by the 81-node program, including selected-width, attention, and both MLP projections, at 384 and 512 bits. Outputs remain synthetic and no scientific result is opened.
- **GO (observation-only):** the native row-batched LayerNorm ABI was exercised with all four distinct actual gamma/beta pairs used by the program at both precisions. The row-batched native GELU ABI is bit-identical to individually dispatched exact GELU jets and is exercised at both program widths 10 and 3072 with deterministic synthetic Jet2 inputs and the closed GPT-2 GELU constants.
- **GO (observation-only):** the compiled exact-fusion ABI was exercised at the actual width 768 for all four branch pairs at both precisions. A liveness-derived, no-static-cache kernel-only composition accounts for every affine, LayerNorm, GELU, per-head attention aggregate, and fused-contrast call required by the 81-node program, but is deliberately not treated as end-to-end timing or a formal bound.
- **GO (correctness-only, disabled by default):** the all-head native causal-attention ABI is bit-identical to concatenated per-head ABI outputs at both precisions. A seven-repetition, alternating-order actual-shape observation found no stable speed advantage, so the resident dispatcher deliberately retains the per-head path.
- **BLOCK:** the full 81-node native resident dispatcher is not implemented. Node orchestration and the static-row cache still live in Python, static past-K/V values are reused only within a cell rather than hash-closed as resident-plan inputs, attention still crosses FFI per head, residual/scalar orchestration remains in Python, and the fusion payload still uses a canonical-JSON control plane pending a native loader.
- **BLOCK:** no numeric replacement cap is authorized. The observations are not formal wall-time bounds and exclude set/copy/comparison/serialization and full certificate orchestration.
- **BLOCK:** real certificate, development rows, confirmation rows, and scientific outcome access remain unauthorized.

## Closure identities

- Fusion-closed TensorProgram semantic hash: `38f40999524d465b8ee58fcc8d2d1822caf9af6c36897a72bd404a8fff34fe62`.
- Exact final-contrast fusion hash: `bd734f457bd3baee252af47f1c048dbd606ec15bf6a1b6533751c7bb943319c1`.
- Native one-cell ordered kernel tags: 81 events; FNV regression checksum `e0f23d0f4c4df894`. The complete 81-tag vector, not the FNV checksum alone, is checked.
- Full regression: 451 tests passed in 222.23 s on the server with the compiled 384/512-bit backend after packed input consumption, batch affine and GELU dispatch, sparse row liveness, static-row reuse, resident fused contrast, and the correctness-qualified/disabled all-head attention ABI.
- Successful actual-dispatch trace SHA-256 at both precisions: `6854f99c2a270b296bac6c1b1ed5ad34d6e18534611e2ef2af5df8e4fa6ff528`.
- Packed resident-plan semantic hash: `0d5625e2f7af118615497e9642481946aec0a436b900e3c0d1661f90ba6f9acf`.
- Packed blob: 28,517,632 bytes; SHA-256 `34bcd45371c08720c23f66d8f723dfc0249779e9e47eee5499c04d6064dc3560`.

## Engineering observations (not formal upper bounds)

- One 384-bit synthetic resident cell: observed maximum 20.3042 s.
- One 512-bit synthetic resident cell: observed maximum 23.3032 s.
- Guardbanded observed maximum for two mandatory cells at both precisions: 109.0184 s.
- 64 synchronized physical-core workers: 1.6537 cells/s at 384 bits and 1.4153 cells/s at 512 bits.
- Fusion-closed static startup guardbands: program parse 0.0091 s, tensor-store full-blob validation 0.1761 s, unique tensor decode/hash 1.6998 s, exact contrast fusion 0.8946 s.
- Peak RSS observed with program/store/fusion validation: approximately 494,136 KiB.
- Across all eight distinct actual affine parameter pairs, observed maxima at 384/512 bits range from 0.01/0.02 s for the 768-by-10 selected projection through 1.79/2.08 s for the 3072-by-768 MLP output projection. These isolated synthetic-input measurements are not formal bounds.
- Across all four actual LayerNorm parameter pairs, observed maxima are 0.04--0.05 s at 384 bits and 0.04--0.06 s at 512 bits. Actual-width row-batched GELU medians are 0.00073/0.00083 s at width 10 and 0.2239/0.2606 s at width 3072 for 384/512 bits; maxima include a 0.0104 s width-10 outlier at 512 bits and 0.2340/0.2940 s at width 3072.
- Actual-width 768 exact fused-contrast medians are 0.0140 s at 384 bits and 0.0161 s at 512 bits; observed maxima are 0.0145 s and 0.0167 s.
- Actual-shape attention medians over seven alternating-order repetitions (`S=12`, 12 heads, head dimension 64): all-head/per-head 0.4842/0.4923 s at 384 bits and 0.4788/0.4707 s at 512 bits. Median speedup ratios 1.0167 and 0.9830 point in opposite directions; no stable speedup is claimed.
- Exact backward liveness for the actual-shape 81-node program materializes 292 of 532 dense axis-0 row slots before static-row-cache savings. Summing isolated per-call medians over those no-cache calls yields 41.3027 s at 384 bits and 47.7685 s at 512 bits; summing 1.25x observed per-kernel maxima yields 54.2361 s and 61.7181 s. This composition excludes scatter/constant materialization, residual and branch arithmetic, cache lookup, Python dispatch, loading, endpoint/center and multi-radius orchestration, adaptive queue/curvature work, certificate serialization, and process-tree RSS; it is therefore neither an end-to-end estimate nor a wall-time upper bound.
- Current deterministic tiny-fixture resident Python-dispatch observations after packed batch affine, row-batched GELU, exact row liveness, static-row caching, and fused contrast: 25.6729 s at 384 bits and 25.4552 s at 512 bits. Repeated development runs varied by several seconds, so no end-to-end speedup is claimed from a single run. These fixture timings are engineering comparisons, not full GPT-2 cell bounds.

## Remaining minimum gate

The next authorized work is outcome-blind implementation and measurement of the remaining performance-resident TensorProgram nodes and a native plan/dispatch entry point, including hash-closed static past-K/V payloads, followed by endpoint/center passes, multi-radius orchestration, adaptive queue, curvature integration, certificate serialization, and process-tree peak memory. Only after those components are hash-closed may a replacement numeric cap be proposed. A measured guardband must not be described as a formal runtime upper bound.
