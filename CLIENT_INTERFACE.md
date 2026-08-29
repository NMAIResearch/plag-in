# PLAG IN client interface

Purpose: let a UI, harness or application use PLAG IN through a stable protocol while keeping all workflow responsibility on the client side.

## 1. Responsibility boundary

| PLAG IN owns | Client or harness owns |
|---|---|
| local model discovery and exact identity | user interface and user accounts |
| inference-engine process lifecycle | system prompts and conversation state |
| local API authentication | agent loops and stopping rules |
| local queue and resource admission | tool definitions and tool execution |
| network binding and locality evidence | MCP clients and servers |
| capability reporting | retrieval and document indexing |
| metadata-only inference receipts | memory and personalisation |
| typed transport and engine failures | approval of consequential actions |

PLAG IN transports tool-call fields when the selected model and endpoint support them. It never executes a tool. It does not decide whether a tool call is safe, correct or authorised.

## 2. Connection contract

A client needs:

- supported wire protocol;
- base URL;
- scoped local API key;
- model alias.

Example values:

```text
base_url=http://127.0.0.1:<port>/v1
api_key=<scoped-local-key>
model=<authorised-alias>
```

The current tested wire protocol is the Chat Completions v1 subset. The setup menu can start the bounded harness gateway and display a vendor-neutral endpoint card. A non-loopback address requires authentication and an explicit serving mode.

## 3. Standard API surface

Mandatory MVP endpoints:

- `GET /health` for process liveness;
- `GET /ready` for selected-engine readiness;
- `GET /v1/models` for authorised aliases;
- `POST /v1/chat/completions` for the common harness path;
- `POST /v1/responses` for the bounded text-only Responses subset in section 3.1;
- `GET /plag-in/v1/capabilities` for measured interface and model capabilities;
- `GET /plag-in/v1/status` for operator state;
- `GET /plag-in/v1/receipts/{request_id}` for metadata receipts.

Capability-gated endpoints:

- `POST /v1/completions`;
- `POST /v1/embeddings`;
- `POST /v1/messages`.

An endpoint is marked `tested` only when every authorised served alias reports that endpoint as tested by its adapter. A mixed set reports the weakest shared evidence state. PLAG IN returns a typed unsupported-capability error instead of silently translating an untested field.

Both request protocols reach one generation path. Neither is ranked above the other in the capability document, the status document or the interactive menus, and neither reads the client's name, user-agent or executable to decide how a request is handled. Each publishes its exact implemented subset under `protocol_subsets` in the capability document, so a client learns what is supported without probing for it by failure.

### 3.1 Responses subset

`POST /v1/responses` implements a bounded text-only subset. It supports `model`, `input` as a string or as message items whose content parts are text, `instructions`, `stream`, `max_output_tokens`, `temperature`, `top_p` and `seed`. The `developer` role is carried as `system`. Output is one message item containing one `output_text` part.

Streaming emits `response.created`, `response.in_progress`, `response.output_item.added`, `response.content_part.added`, `response.output_text.delta`, `response.output_text.done`, `response.content_part.done`, `response.output_item.done` and `response.completed`, each numbered in sequence. Generation completes before the first event is written, so the text arrives as one delta. This is client-protocol compatibility, not token-by-token streaming.

The following are refused with a typed unsupported-capability error naming the field, never silently ignored: `tools`, a `tool_choice` other than `none`, `background`, `store`, `previous_response_id`, `conversation`, `include`, `reasoning`, `truncation`, `text` format selection, `parallel_tool_calls`, `prompt`, `attachments`, non-message input items, and `input_image`, `input_file` or `input_audio` content parts.

PLAG IN executes no tool, runs no agent loop and keeps no conversation state on either protocol. Every response states `tools: []`, `tool_choice: "none"`, `store: false` and `parallel_tool_calls: false` rather than leaving a client to infer them from a silent success. Authentication, alias scope, body-size limits, body-read timeout, client-disconnect cancellation and receipt binding are the same for both protocols, because both use the same request handler and the same admission path.

A request-body limit of 1,000,000 bytes applies to every route. An oversized request receives a typed `payload_too_large` refusal. That refusal is sent before the body is read and its delivery is best effort: the connection may still fail while the client is writing.

## 4. Streaming and tools

The MVP accepts `stream: true` on Chat Completions and returns a buffered server-sent event response after generation completes. It emits the completed assistant message as a chat-completion chunk, emits the terminal finish state, then emits `[DONE]`. Message fields, including tool-call fields, retain their response order. This is client-protocol compatibility, not token-by-token streaming, and the capability document reports `tested_buffered`.

The embedded text route checks client liveness during generation, supplies the same cancellation state to the native abort callback and applies a finite generation-time bound. Cancelled work records failure rather than completed inference. A later token-by-token route must preserve event order, termination state, tool-call fields and backend errors before it can replace the buffered route.

The client supplies tool schemas. PLAG IN validates only the transport envelope required by the declared API subset. Tool-call arguments remain untrusted client-side data.

## 5. Capability document

The capability document is machine-readable and versioned. It includes:

- PLAG IN protocol version;
- supported wire formats and endpoints;
- the exact implemented subset of each request protocol, under `protocol_subsets`;
- authorised model aliases;
- per-model capability state as `tested`, `declared`, `inferred` or `unknown`;
- per-model compatibility state as `tested`, `unverified` or `unsupported`, under `model_compatibility`;
- what that state was decided on, under `model_compatibility_evidence`: the state, the reason, the reviewed compatibility record identifier, the recorded model SHA-256 and the observed model SHA-256. The same record appears in the status document and in the receipt adapter record. It is derived from the model bytes at registration and is never accepted from a client;
- streaming status;
- tool-call transport status;
- structured-output status;
- vision and embedding status;
- context limit and its evidence state;
- active locality enforcement level;
- receipt support;
- compatibility-test timestamp and fixture digest.

The document does not claim a model capability from its name alone. `model_compatibility` and the status document's `service_class` are read from the profile the operator confirmed when the backend was registered. A request that succeeds never raises either of them, so a model that answers under an unverified trial continues to report itself as an unverified trial.

The Responses endpoint carries the same evidence state as Chat Completions and never a better one, because both reach the same backend generation call. What differs between them is the request and response shape, which `protocol_subsets` states in full.

## 6. Request and receipt binding

PLAG IN assigns a request ID or accepts a valid caller ID. Responses include:

```text
X-PLAG-IN-Request-ID: <id>
X-PLAG-IN-Receipt-ID: <id>
X-PLAG-IN-Locality-Level: L0|L1|L2|L3
```

The receipt associates the request identifier with an exact weight digest and with declared model, engine, gateway and configuration state, under a local HMAC. It is emitted alongside the response. **It does not cryptographically bind response content**: the schema carries no response digest and no authenticated transport transcript, so nothing in the receipt establishes which bytes the client received. The association between a receipt and a response is the request identifier that PLAG IN assigned.

From receipt schema version 3 onwards the receipt also binds the runtime Python source-tree snapshot taken at gateway initialisation and reverified unchanged immediately before receipt persistence, and status reports the same snapshot. Those versions fail closed on an absent or malformed source identity. This is a source-tree snapshot, not proof of executed bytecode, native-library behaviour, build provenance or external attestation, and it is not a content digest: it excludes prompt content and a prompt-derived digest by default.

Receipt schema version 5 is the current write version. Versions 2, 3 and 4 remain readable and are never rewritten.

Status and receipts include the confirmed embedded runtime profile. The profile states that the engine is `libllama_embedded`, that no Ollama runtime dependency exists, and records requested and effective context, GPU offload, thread, batch, micro-batch, parallel-sequence and fallback sampling values. Request-specific sampling remains client-controlled. A transitional `llama_server_direct` adapter reports its own identity and is never presented as the embedded route.

The embedded runtime profile also records the registered native ABI profile. A complete digest match without a registered FFI-to-bundle profile is refused before any native library loads.

## 7. Authentication

The gateway authenticates client applications, not the harness's end users. A client key may be scoped to:

- one or more model aliases;
- permitted endpoints;
- loopback or approved network origin;
- request and queue limits;
- an expiry time.

Keys are displayed once. Configuration exports omit secrets unless the operator requests an explicit secret-bearing output.

## 8. Configuration export

```text
plag-in connect --format manual
plag-in connect --format openai-env
plag-in connect --format env
plag-in connect --format json
plag-in connect --format curl
```

`manual` reports local transport, protocol, base URL, authentication header and model alias without presenting a vendor preset as the gateway boundary. The optional OpenAI-style environment form states that it addresses the local loopback gateway and has no remote provider. These commands print connection material. They do not edit a harness. Harness-specific recipes live outside the core under `integrations/` and contain configuration only.

## 9. Harness evidence order and labels

A protocol route is defined by a generic raw client with an arbitrary user-agent, and only then exercised by a named harness. Named-harness evidence is an integration result about that harness; it is never the definition of the route, and the same pass criteria apply to every harness.

Harness support is reported with these labels and no product-name allowlist:

- `protocol_tested`: the core surface has passed the project conformance suite;
- `integration_tested`: a named harness and version has completed an approved live trial;
- `protocol_compatible_unverified`: the harness uses a supported protocol but has no named live evidence;
- `adapter_required`: the harness requires a different or extended protocol.

A harness needing Anthropic Messages, Google GenAI or another protocol receives `adapter_required` naming the exact missing protocol. It does not receive a harness-name branch in the gateway.

## 10. Odysseus first-client route

Odysseus treats an unknown local host as an OpenAI-compatible endpoint, obtains model IDs from `/v1/models` and requests Chat Completions streaming. PLAG IN supplies the buffered server-sent event compatibility route without importing Odysseus code into the gateway.

The first integration test supplies PLAG IN's base URL, local key and model alias to an Odysseus test configuration. Odysseus remains responsible for messages, tools, agent behaviour, memory and UI. No live Odysseus configuration changes are authorised by this specification.

## 11. Adapter rule

No harness SDK is imported into the gateway core. A client requiring a non-standard protocol receives a separate adapter process or configuration recipe. The adapter has its own tests and cannot weaken the gateway's locality or receipt claims.
