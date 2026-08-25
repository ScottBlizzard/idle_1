# GREEN v1.3.3 prepare terminal record

The unique v1.3.3 prepare stopped at `06E_FIXED_BATCH_GRAPH` before development.
Its output recentering made the zero endpoint bitwise exact but the first
nonzero `x_only` endpoint differed from the independent batch-one full-hook
reference by `4.57763671875e-05`, above the unchanged `2e-5` limit.

Terminal hashes:

- `result.json`: `e1084e999ff3c94c7d7cec343f22b6d7462f142440955edcde561b860d36a1d8`
- `run_ledger.json`: `72d4c9ae7f3af99bb25c1a28c3576ad29772a1585dbc8c304a9b482dfd11a51d`
- `scientific_invariance_v133.json`: `cd4395f49f5073c09a2d20579006ee3c96046c7a855f5cc7353e414491c66297`

The terminal output is hard-link archived at
`/mnt/sdb/ccj/iclr_1_runs/green_bridge_v133_terminal_archive_20260825`.
The prepare-only follow-up diagnostic proved that anchor-relative block-10 plus
recentering still failed, whereas actual batch-one manual endpoints matched all
tested full-hook values and both central derivatives bitwise exactly. The
diagnostic hash is
`7a35dd8d3ad21973850e20466a8d39cb3f0eea4ab73a725617c6c30ea2da0ab5`.

