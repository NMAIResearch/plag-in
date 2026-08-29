# Verification boundary

Purpose: state what has and has not been established for PLAG IN `0.1.0a2`.

## Public source suite

The declared suite passed 791 of 791 tests, N = 791, on 2026-08-29.

It covers configuration validation, identity binding, path containment, compatibility derivation, bounded GGUF parsing, receipt integrity, receipt-store generations, API authentication, body and generation bounds, setup controls, resource-policy paths, embedded-adapter fixtures and failure cleanup.

Fixture engines and native doubles do not establish live model behaviour. A host-context pass is not a restricted-sandbox pass.

## Local model matrix

One outcome was recorded for each of 27 held manifest entries, N = 27:

| Outcome | Count |
|---|---:|
| `live_pass` | 5 |
| `resource_refused` | 20 |
| `duplicate_effective_identity` | 1 |
| `remote_only` | 1 |

Five models loaded sequentially. Each answered the non-streaming, streaming and two-turn requests and released its process, listener and GPU allocation. Only the exact Qwen2.5 3B profile served as `tested`; four remained `unverified`.

`live_pass` is an operational criterion. It does not establish semantic instruction following, general model quality or safety.

## Protocol and client matrix

Raw authenticated HTTP passed:

1. Chat Completions, non-streaming.
2. Chat Completions, streaming with `[DONE]`.
3. Responses, non-streaming.
4. Responses, streaming with nine declared events in fixed order.

Actual clients produced narrower results:

| Client | Result | Observed boundary |
|---|---|---|
| Codex CLI | `adapter_required` | The client sent `tools`; the bounded Responses subset returned typed HTTP 501. |
| Claude Code | `adapter_required` | The client requested the unimplemented Anthropic Messages route. |
| Gemini CLI, legacy | `trial_failed` | The client refused its authentication configuration before issuing a request. |
| Antigravity CLI | `trial_failed` | No route or HTTP status was reported; whether a request reached PLAG IN is unassessed. |
| Odysseus fixture | `protocol_compatible_unverified` | Discovery and path logic were exercised, but `urllib`, not the Odysseus HTTP client, issued the request. |

Raw protocol success is not presented as success for an installed client.

## Receipt boundary

The five live model receipt stores passed a retained chain recheck. The exact tested Qwen receipt carried its compatibility evidence block.

Receipts bind local model and runtime evidence under a local integrity mode. They do not bind response content and do not establish third-party authenticity. The request identifier associates a response and receipt inside the emitting system.

## Unassessed

- Original Phase 2 per-entry resource measurements. Repair-time measurements are recorded separately.
- Runtime behaviour of the 20 refused models.
- Effective GPU layer offload.
- First-token latency and comparative performance.
- Calibration of the provisional working-set formula.
- Historical absence of provider contact or download during the live matrix.
- Complete external non-overlap evidence for the original, superseded harness runs.
- Odysseus application and HTTP-client integration.
- Google GenAI protocol behaviour.
- Antigravity behaviour through its undocumented custom-model structure.
- Claude Code behaviour beyond the missing Messages route.
- Codex CLI behaviour beyond the unsupported `tools` field.
- Windows and macOS behaviour.
- Public-network or multi-user deployment.
- Security of model weights or the native loader.
- Browser-origin, DNS-rebinding and per-client least-privilege controls.

## Reproduce

Run in your terminal:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests -t .
```

Run the release boundary check in a Git checkout:

```text
python3 scripts/release_gate.py
```

Some tests require loopback socket creation. Record sandbox refusal separately rather than converting it into a product failure.
