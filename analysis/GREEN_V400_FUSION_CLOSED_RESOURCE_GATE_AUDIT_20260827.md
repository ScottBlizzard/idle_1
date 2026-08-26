# GREEN v4.0.0 fusion-closed resource-gate audit — 2026-08-27

## Decision

- **GO:** the outcome-blind GPT-2 joint-witness TensorProgram is closed at 81 nodes, exact zero control, fixed softmax pivot 0, final unpadded causal position, element dependency masks, tensor-store references, semantic dispatcher signature, and exact final-contrast fusion.
- **GO:** the frozen 100,000,000 MPFR-operation cap is infeasible. The dense coefficient-arithmetic lower bound is 341,397,504 operations per precision/cell at sequence length 12; two mandatory initial cells already require at least 682,795,008 at one precision.
- **GO (narrow scope):** the native synthetic resident proxy executes the same ordered kernel-tag vector as the 81-node program and counts exactly 352,275,450 directed-enclosure arithmetic primitives per precision/cell under the frozen taxonomy.
- **BLOCK:** no numeric replacement cap is authorized. The observations are not formal wall-time bounds and exclude set/copy/comparison/serialization and full certificate orchestration.
- **BLOCK:** real certificate, development rows, confirmation rows, and scientific outcome access remain unauthorized.

## Closure identities

- Fusion-closed TensorProgram semantic hash: `38f40999524d465b8ee58fcc8d2d1822caf9af6c36897a72bd404a8fff34fe62`.
- Exact final-contrast fusion hash: `bd734f457bd3baee252af47f1c048dbd606ec15bf6a1b6533751c7bb943319c1`.
- Native one-cell ordered kernel tags: 81 events; FNV regression checksum `e0f23d0f4c4df894`. The complete 81-tag vector, not the FNV checksum alone, is checked.
- Full regression: 443 tests passed on the server with the compiled 384/512-bit backend.

## Engineering observations (not formal upper bounds)

- One 384-bit synthetic resident cell: observed maximum 20.3042 s.
- One 512-bit synthetic resident cell: observed maximum 23.3032 s.
- Guardbanded observed maximum for two mandatory cells at both precisions: 109.0184 s.
- 64 synchronized physical-core workers: 1.6537 cells/s at 384 bits and 1.4153 cells/s at 512 bits.
- Fusion-closed static startup guardbands: program parse 0.0091 s, tensor-store full-blob validation 0.1761 s, unique tensor decode/hash 1.6998 s, exact contrast fusion 0.8946 s.
- Peak RSS observed with program/store/fusion validation: approximately 494,136 KiB.

## Remaining minimum gate

The next authorized work is outcome-blind implementation and measurement of the actual resident TensorProgram dispatcher, endpoint/center passes, multi-radius orchestration, adaptive queue, curvature integration, certificate serialization, and process-tree peak memory. Only after those components are hash-closed may a replacement numeric cap be proposed. A measured guardband must not be described as a formal runtime upper bound.
