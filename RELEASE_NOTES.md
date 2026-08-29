# PLAG IN 0.1.0a2

Purpose: describe the second public source-alpha payload and its verified boundary.

## Included

- General admission for already-held local GGUF models through explicit unverified trials.
- Exact reviewed compatibility records bound to model bytes and a registered native ABI profile.
- A bounded GGUF header, metadata and tensor-descriptor reader with no new dependency.
- Authenticated Chat Completions and Responses text subsets.
- Typed refusal for unsupported tools, images, files, background mode and persistent conversation state.
- Content-free receipt stores with schema generations, authenticated checkpoints and an independent verifier.
- State-file confinement, active-store rollback, body limits, generation bounds and cleanup regressions.
- Product and client-interface specifications.
- Unit and integration tests for the published source.

## Verification

The public suite passed 791 of 791 tests, N = 791, on 2026-08-29.

The local model matrix contained 27 manifest entries, N = 27: 5 `live_pass`, 20 `resource_refused`, 1 `duplicate_effective_identity` and 1 `remote_only`.

The five live trials loaded sequentially, answered the declared requests, produced receipts and released their listener and GPU allocation. Only the exact registered Qwen2.5 3B profile served as `tested`. The other four remained `unverified`.

Raw authenticated HTTP passed both implemented request protocols in streaming and non-streaming form. Actual client evidence remains bounded:

- Codex CLI and Claude Code are `adapter_required` for unsupported protocol semantics.
- Legacy Gemini CLI and Antigravity CLI are `trial_failed`.
- Odysseus is `protocol_compatible_unverified`, not integration tested.

## Material limits

1. The working-set formula is provisional and refused 20 held entries under the declared trial cgroup.
2. Refused models were not loaded, so their runtime behaviour is unassessed.
3. Effective GPU layer offload remains unassessed.
4. No comparative latency, throughput or memory baseline was run.
5. Original Phase 2 per-entry resource measurements were not retained. The released matrix used a separate repair-time measurement.
6. Historical external traffic absence is unassessed because no capture was retained.
7. The operational live-pass criterion does not establish semantic instruction following or model quality.
8. The native loader remains in the gateway process. Valid metadata does not establish that model bytes or native libraries are safe.
9. Per-client key scopes, explicit browser-origin controls and Host-header hardening are not part of this version.
10. Linux is the only supported platform.

## Packaging boundary

- Source distribution only.
- No model weights, native runtime, CUDA library or installer included.
- No internal handoff, host-specific trial script, local state or working evidence trail included.
- No tag or packaged GitHub release is implied by a source update to `main`.

## Licence

Apache License 2.0.
