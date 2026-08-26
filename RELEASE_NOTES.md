# PLAG IN 0.1.0a1

Purpose: describe the public source-alpha payload and its verified boundary.

## Included

- Python source for the PLAG IN command-line interface and gateway.
- Embedded `libllama` adapter and transitional direct-worker adapter.
- Local model inventory, identity and containment controls.
- Registered native ABI-profile enforcement.
- Content-free schema-v3 receipts and chain verification.
- Confirmed setup, local test-chat and harness-connection paths.
- Unit and integration fixtures.

## Verification

The complete fixture suite passed 311 of 311 tests, N=311, on 2026-08-26.

One bounded live L3 trial completed one synthetic request, N=1 request. The response completed, one schema-v3 receipt was written, the chain validated and post-run process, listener and GPU allocation checks found no trial residue.

The accepted result carries three material limits:

1. The systemd wrapper read `Result=success` after collection. A control query showed that this value is vacuous for a missing unit. Journal evidence, not that assertion, supports clean scope completion.
2. The packet capture ran inside a network namespace exposing loopback only and filtered loopback traffic. Its zero-packet result carries little evidence beyond the namespace and failed external-connectivity probe.
3. Transient execution observations were preserved in the command transcript rather than a separate immutable execution log.

The trial does not establish production readiness, repeated-run reliability, model quality, exact GPU layer offload or comparative performance. No baseline was run.

## Platform and packaging limits

- Linux only.
- Python 3.14 or newer.
- Source distribution only.
- No native runtime, model weights or installer included.
- Confirmed serving requires the exact registered native bundle and already-held model.
- Ordinary operation does not install an operating-system egress policy.

## Selected publication target

- Licence: Apache License 2.0.
- Repository: `NMAIResearch/plag-in` on GitHub.

## Remaining publication blockers

- Authenticate the selected hosting account.
- Build and inspect the final source archive.
- Re-run the suite against the exact release commit.
- Obtain explicit approval for the final remote target and payload.
