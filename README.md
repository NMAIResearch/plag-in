# PLAG IN

**Your model. Your device. No outbound telemetry. Verifiable local inference.**

PLAG IN is an experimental source gateway for direct, inspectable access to local GGUF models. The statement above is the product requirement. It is not a blanket privacy guarantee.

Version `0.1.0a2` supports Linux and Python 3.14 or newer. It does not download models, install a native runtime, execute tools or silently fall back to a remote provider.

## Current evidence

The public source passed 791 of 791 fixture and integration tests, N = 791, on 2026-08-29.

A bounded local compatibility exercise recorded one outcome for each of 27 held manifest entries, N = 27:

- 5 completed the declared live operational trial;
- 20 were refused by the declared resource policy before loading;
- 1 was a duplicate effective identity;
- 1 required a remote provider and was not contacted.

Only the exact registered Qwen2.5 3B profile served as `tested`. Four other models that completed the trial remained `unverified`. A successful response did not promote a model.

The operational trial checked load, model listing, non-streaming and streaming generation, a two-turn template, receipts and cleanup. It did not establish model quality. No comparative performance baseline was run.

Raw authenticated HTTP passed the implemented Chat Completions and Responses text subsets in streaming and non-streaming forms. Installed client results were narrower:

- Codex CLI required unsupported Responses tool semantics and was `adapter_required`.
- Claude Code required the Anthropic Messages API and was `adapter_required`.
- Legacy Gemini CLI and Antigravity CLI were `trial_failed`; no Google GenAI protocol result was observed.
- An Odysseus discovery and path fixture was `protocol_compatible_unverified`; its own HTTP client and running application were not exercised.

Historical absence of external traffic during these compatibility trials is unassessed because no traffic capture was retained. Ordinary PLAG IN operation does not install an operating-system egress policy.

## What it provides

- Exact local model-file hashing and path containment.
- Bounded GGUF header and metadata inspection before native load.
- Registered native ABI profiles bound to complete component digests.
- Exact compatibility records for reviewed model and native-profile pairs.
- Explicit unverified local trials without automatic promotion.
- Embedded `libllama` inference with no child inference server.
- Authenticated loopback Chat Completions and Responses text subsets.
- Requested and effective runtime-profile records.
- Content-free, hash-chained receipts with active-store generations and verification.
- Bounded body, generation, memory and swap controls.
- Read-only host inspection and a confirmed first-run workflow.

PLAG IN is not a model distributor, chat application, agent framework, browser security boundary or substitute for operating-system isolation.

## Install from source

Run in your terminal:

```text
python3 -m venv .venv
```

Then run in your terminal:

```text
.venv/bin/python -m pip install .
```

The source installation may obtain its declared build dependency if it is absent. It never installs a model.

Inspect the host without writing configuration or starting a model:

```text
.venv/bin/plag-in doctor --format json
```

Open the interactive setup:

```text
.venv/bin/plag-in
```

The menu uses the arrow keys and Enter. Configuration replacement, receipt-store creation and native model loading each require confirmation.

## Model admission

Availability, compatibility and resource admission are separate results.

A held local GGUF may be offered as an unverified trial when its bytes resolve safely and the declared resource policy permits a load. This is not a claim that every GGUF architecture, quantisation or template will work.

The exact registered Qwen2.5 3B profile carries reviewed compatibility evidence. A configuration string cannot create that evidence. The serving path re-derives compatibility from the bytes resolved for the load.

Model self-description is not identity evidence. PLAG IN reports the local alias, complete model digest and recorded metadata instead.

## API boundary

PLAG IN exposes authenticated loopback routes for:

- `/v1/models`;
- a bounded `/v1/chat/completions` text subset;
- a bounded `/v1/responses` text subset;
- status and capability documents used by the local operator flow.

Unsupported features are refused by name rather than silently ignored. PLAG IN does not execute tools or create an agent loop.

The current alpha uses a generated local API key. Per-client identities and least-privilege key scopes are not implemented in `0.1.0a2`. Browser-origin and Host-header hardening are also not claimed.

## Verify

Run in your terminal:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests -t .
```

Some tests bind loopback ports. A sandbox that denies socket creation cannot run those tests.

Verify a receipt store:

```text
python3 tools/verify_inference_receipts.py --help
```

See [product specification](PRODUCT_SPEC.md), [client interface](CLIENT_INTERFACE.md), [verification boundary](docs/VERIFICATION.md), [security policy](SECURITY.md) and [release notes](RELEASE_NOTES.md).

## Licence

Licensed under the Apache License 2.0. See [LICENSE](LICENSE).
