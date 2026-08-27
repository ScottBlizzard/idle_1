# GREEN v4.0.0 fusion-closed resource-gate audit — 2026-08-27

## Decision

- **GO:** the outcome-blind GPT-2 joint-witness TensorProgram is closed at 81 nodes, exact zero control, fixed softmax pivot 0, final unpadded causal position, element dependency masks, tensor-store references, semantic dispatcher signature, and exact final-contrast fusion.
- **GO:** the frozen 100,000,000 MPFR-operation cap is infeasible. The dense coefficient-arithmetic lower bound is 341,397,504 operations per precision/cell at sequence length 12; two mandatory initial cells already require at least 682,795,008 at one precision.
- **GO (narrow scope):** the native synthetic resident proxy executes the same ordered kernel-tag vector as the 81-node program and counts exactly 352,275,450 directed-enclosure arithmetic primitives per precision/cell under the frozen taxonomy.
- **GO (correctness scope):** the actual Python MPFR TensorProgram dispatcher emits all 81 semantic events only after successful node completion at both 384 and 512 bits. The event vector matches the program dispatcher signature and all five compiled/reference roots remain bit-identical on the deterministic tiny fixture.
- **GO (prepare-only scope):** the actual GPT-2 constants needed by the 81-node program are closed into one immutable 64-byte-aligned packed blob: 32 resident tensor records plus four exact-fusion source tensors cover all 36 unique program tensor references and all 150 node/input bindings. The packed plan is validated but remains explicitly `native_execution_ready=false`.
- **GO (narrow resident-fusion correctness scope):** the compiled dispatcher consumes the plan's exact fused final contrast without loading the four full-unembedding source tensors; all five roots remain bit-identical to the reference dispatcher at 384 and 512 bits on the deterministic tiny fixture.
- **GO (observation-only):** the native packed affine ABI was exercised with the actual `block11.mlp.W_in` 768-by-3072 weight and synthetic Jet2 inputs. Its outputs are repeatable and its smaller regression fixture is bit-identical to individually dispatched exact affine columns at both precisions.
- **BLOCK:** the full 81-node native resident dispatcher is not implemented. Most nodes still pass through the Python JSON/FFI correctness dispatcher, static past-K/V caches are not resident-plan inputs, and the fusion payload still uses a canonical-JSON control plane pending a native loader.
- **BLOCK:** no numeric replacement cap is authorized. The observations are not formal wall-time bounds and exclude set/copy/comparison/serialization and full certificate orchestration.
- **BLOCK:** real certificate, development rows, confirmation rows, and scientific outcome access remain unauthorized.

## Closure identities

- Fusion-closed TensorProgram semantic hash: `38f40999524d465b8ee58fcc8d2d1822caf9af6c36897a72bd404a8fff34fe62`.
- Exact final-contrast fusion hash: `bd734f457bd3baee252af47f1c048dbd606ec15bf6a1b6533751c7bb943319c1`.
- Native one-cell ordered kernel tags: 81 events; FNV regression checksum `e0f23d0f4c4df894`. The complete 81-tag vector, not the FNV checksum alone, is checked.
- Full regression: 446 tests passed on the server with the compiled 384/512-bit backend after the packed-plan, packed-affine, and resident-fused-contrast additions.
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
- Actual packed `block11.mlp.W_in` affine observed maxima: 1.4126 s at 384 bits and 1.6704 s at 512 bits; 1.25x observational guardbands 1.7657 s and 2.0880 s, respectively. These are not formal bounds and are not a complete resident-cell runtime.

## Remaining minimum gate

The next authorized work is outcome-blind implementation and measurement of the remaining performance-resident TensorProgram nodes and native plan loader, including the static past-K/V closure, followed by endpoint/center passes, multi-radius orchestration, adaptive queue, curvature integration, certificate serialization, and process-tree peak memory. Only after those components are hash-closed may a replacement numeric cap be proposed. A measured guardband must not be described as a formal runtime upper bound.
