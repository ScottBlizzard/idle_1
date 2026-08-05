# GREEN v1.3 pre-implementation hash corrigendum request — 2026-08-06

Binding decision reviewed:

```text
analysis/GPTPRO_GREEN_GATE08_V12_DECISION_20260805.md
SHA-256 e556a1fdc282f513288e13047cc68415f197dcf15effb9e947eb0169c7394d60
```

The v1.3 archive procedure was executed only through its read-only verification
step. It stopped before moving or modifying any artifact because the binding
decision contains a malformed historical `manifest.json` SHA-256.

The decision repeats this 63-character value in the historical artifact table
and archive verifier:

```text
ea486fe8eea798b16951fcea9394b1c4ddb4b44bd4afb5c8b104b37aaf047be
```

The preserved server artifact has this actual 64-character SHA-256:

```text
ea486fe8eea798b16951fcea9394b1c4ddb4b44bbd4afb5c8b104b37aaf047be
```

The missing character is one `b` in the segment:

```text
document: ...c4ddb4b44bd4...
actual:   ...c4ddb4b44bbd4...
```

The other four binding archive hashes were verified exactly:

```text
390c5b62d5b42e216abbb15a0d6d206a55419c48117f610f34c0ac802e153747  result.json
49aa7a1818fb06d63b975938aea7285d3198fccc97723a96a37afa097abdbb99  hook_audit.json
2c8dd401b93d3864969ab941b85cae2ab5e6e983bdf39b909f33c532b480cc16  donor_v2_plan.json
fa88911fcce749942a24c9e479c66cf89cd72ce9386b76d146262de6671b4f65  run_ledger.json
```

Server state after the failed verification:

- branch `main`;
- HEAD `da5161ad2e87a9bfb7de8bf772af7969ff531f64`;
- clean status porcelain;
- `outputs/green_bridge` remains present and unchanged;
- the requested v1.2 archive directory was not created;
- no v1.3 code was implemented;
- no model or scientific endpoint was run.

This is a provenance transcription inconsistency, not a scientific result and
not an authorization to edit the binding decision locally. GPTPro must issue a
short binding corrigendum that specifies whether every active v1.3 archive,
manifest, and contract-test reference must use the actual 64-character hash.
