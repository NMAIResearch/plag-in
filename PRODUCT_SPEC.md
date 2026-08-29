# PLAG IN product specification

Purpose: state the minimum useful product and prevent a small gateway from expanding into another harness.

## 1. Primary user

A person or small organisation operating one workstation or a small private GPU pool, with local model files already present or a need for bounded model recommendations.

The initial platform is Linux. Other platforms remain out of scope until the Linux acceptance suite passes.

Market demand is an unverified hypothesis. The private prototype tests operational usefulness, not demand or willingness to pay.

## 2. Required user journeys

### 2.0 First use

```text
plag-in
plag-in doctor
```

In an interactive terminal, a no-argument invocation presents a persistent menu navigated with Up, Down and Enter. Numbered input remains the fallback for buffered interfaces and tests. Inspection is read-only. Configuration and private test chat choices preview their effects and require separate confirmation. `doctor` performs the inspection directly and supports human or JSON output.

The first-use path must:

- state before inspection that inspection performs no network action, configuration write or model start;
- return to the menu after every completed choice until the user selects Exit;
- list held Ollama model tags and manifest-declared sizes without hashing every weight file, while labelling untested models as `held, unverified`;
- configure only an exact compatibility-tested held model and native bundle;
- preview the target path before writing configuration and require confirmation before replacement;
- preview model identity, runtime, locality controls and receipt path before a private test chat;
- detect a pre-current receipt record before model loading, offer a separate current-schema store with explicit confirmation, and preserve the historical store byte-identically;
- verify a hard process memory ceiling and immediate host resource headroom before model loading;
- stop the listener and model when the private test chat closes;
- offer either terminal chat or a vendor-neutral connector for an existing harness;
- report current platform support without implying Windows or macOS compatibility;
- keep locality controls separate from runtime settings;
- classify configuration inspection as no more than L0;
- show bounded next steps after each selection;
- refuse interactive setup in a non-interactive process;
- preserve explicit subcommands for scripts and harnesses.

Locality controls cover binding, authentication, external routes, remote fallback, operating-system network policy, traffic observation and content logging. Runtime settings cover context, GPU offload, threads, batching, parallel sequences and sampling. A runtime value never changes the locality evidence level.

### 2.1 Existing model

```text
plag-in inspect
plag-in serve <model-alias>
```

By default `inspect` inspects only the models declared by the configured profiles: it resolves and hashes exactly those profile models under the same containment and Ollama manifest-to-blob identity checks as `serve`, and does not enumerate or hash unrelated held models. The explicit `--all-models` option requests the full inventory scan of every held GGUF and Ollama model under the configured roots. Both outputs state their scope (`configured_profiles` or `all_models`).

The setup menu uses a separate quick inventory for model selection. It reads manifest identity, declared size and blob presence without hashing the weight bytes. This quick inventory is visibility evidence only. Selection does not promote an unverified model into the compatibility-tested set. `inspect --all-models` remains the explicit complete-hash operation.

`inspect` reports:

- discovered inference runtimes and versions;
- for each configured profile, the resolved weight location and hash, or an explicit `unassessed` status with its reason when the configured path is missing or uncontained;
- discovered model manifests and weight locations under `--all-models`;
- weight hashes or `hash pending`;
- model format and quantisation where deterministically readable;
- tested and untested capabilities;
- estimated memory requirement, labelled as an estimate;
- current port and process conflicts.

`serve` must:

- resolve an exact model identity before launch;
- present the embedded `libllama` runtime profile and require confirmation before acquiring a listener or loading model weights;
- make context size, GPU offload, CPU threads, batch size, micro-batch size, parallel slots and server sampling fallbacks explicit;
- show the selected runtime, weights, locality mode and listening address;
- require confirmation if a requested client binding leaves loopback for an approved private network;
- launch an argument array without a shell;
- bind to loopback by default;
- refuse unauthenticated non-loopback binding;
- write a session receipt.

### 2.2 Missing model

```text
plag-in recommend
```

The command asks for task, available memory, preferred speed or quality, context requirement and licence constraint. It returns no more than three options.

Each option records:

- exact model and quantisation;
- official or declared distributor;
- licence and any unresolved licence field;
- download size;
- expected memory requirement, labelled as an estimate;
- required engine;
- verified download command or `unverified`;
- expected content hash when the distributor publishes one.

Recommendation never downloads. Download is a separate explicit command and must show destination, expected size, source and hash policy before execution.

### 2.3 Verify locality and reproducibility

```text
plag-in verify
plag-in receipt <request-id>
```

`verify` checks the active enforcement level, network binding, configured local routes, running child process, engine version, model hash status and logging policy. It must distinguish a measured control from a declared setting.

The receipt associates a request identifier with an exact weight digest and optional declared source and runtime metadata under a local HMAC. It is emitted alongside a response. It does not cryptographically bind response content: the schema carries no response digest and no authenticated transport transcript, so nothing in the receipt establishes which bytes were returned to the client.

The receipt records:

- request ID and UTC timestamps;
- gateway and native runtime versions;
- model alias and weight digest;
- template and configuration digests;
- locality enforcement level;
- selected route;
- listening and backend addresses;
- input and output token counts when reported by the engine;
- latency and terminal status;
- the runtime source digest and its explicit scope label, from receipt schema version 3 onwards, binding the runtime Python source-tree snapshot taken at gateway initialisation and reverified unchanged immediately before receipt persistence; this is a source-tree snapshot, not proof of executed bytecode, native-library behaviour, build provenance or external attestation;
- whether prompt or response content was retained.

Prompt and response content are not retained by default.

Receipt schema version 5 is the current write version. Versions 2, 3 and 4 remain readable under their documented historical integrity rules and are never rewritten. A store holding a pre-version-5 record refuses a version 5 append and names the operator action:

```text
plag-in receipt --initialise-current-store
```

That command creates a current-schema store beside the existing one, leaving the existing store and its checkpoint byte-identical. Both paths and their roles are reported by `plag-in status` and by every `plag-in receipt` check.

Which store is active is recorded in a pointer file in the state directory. The pointer may name one path only: a regular file, not a symbolic link, directly beneath that state directory, under the current-schema filename, declaring the current schema version. Any other value, a symbolic link at the pointer or at the store, a pointer naming a store that does not exist, and a pointer that cannot be parsed are all refused. The product reports the refusal rather than falling back to the earlier store, because falling back would write to a different store from the one the operator activated without saying so. Initialisation and activation are one transaction from the operator's side: a failure at either step leaves no store, key, checkpoint, pointer, temporary file or staging directory behind. The lock that serialises PLAG IN processes covers initialisation and rollback. Writing the pointer that activates the store happens between those two operations and is not under that lock; it is made safe by the pointer rules above rather than by serialisation. Two zero-byte lock files may remain and are never removed: the transaction lock itself, and the store's own lock, which another process may be holding open. Removing a lock that a second process holds would split later users of that store across two lock inodes, which is worse than an empty file left in place.

Both lock names are confined on the same terms as the store: a symbolic link, a FIFO or a device node at the transaction lock or at a store's own lock is refused, and nothing is created at the far end of it. The transaction lock is taken before the store, key and checkpoint names are examined, so it is the first name in the transaction that anything can reach, and leaving it unconfined admitted at the lock what the pointer rules refuse at the store.

The store file and its HMAC key are confined at the point of opening, whether or not the name is already occupied. A link at either name is refused, no file mode outside the state directory is changed, and no outside bytes are accepted as a key. These names are reachable without the pointer: until a current-schema store is activated, the product opens the original store and key directly, so their confinement cannot rest on the pointer rules.

Anything at those names that is not a regular file is refused as a typed error, and the refusal does not wait. A directory, a device node and a FIFO are each rejected on the same terms as a link. This is stated because a FIFO is the case where a confinement check can be defeated without being defeated: opening one waits for a peer at the other end, so a check that runs after the open never runs at all. The open is non-blocking until the file is known to be a regular file, and a name that would have waited is refused instead.

The same terms cover the store's checkpoint. It is read through a confined descriptor rather than by name, because it is the anchor that records how many records should exist: bytes planted at that name would otherwise make an empty store report as sound, and a FIFO there would hold a verification open rather than fail it. A checkpoint is bounded at 4096 bytes. That is a maximum size and not a maximum read: a larger file is refused rather than accepted on the strength of its first 4096 bytes. Reads accumulate until the file ends or the bound is exceeded, because a read that returns fewer bytes than it was asked for is permitted and is not evidence that the file ends there.

Every product route that reads the receipt set applies these terms, the conformance check included. That check also takes the store's own lock while it reads the log, the key and the checkpoint, so the three come from one snapshot. Without it an ordinary append could complete between the log read and the checkpoint read, and the check would report a mismatch against a store that verifies as sound. The lock is released before the specification rules are assessed: it covers taking a coherent snapshot, not judging it.

Whether the product created a store is established by the creation itself, not by looking at the name beforehand. A store that was already present, including one created by another party between the start of an attempt and its own creation step, is left byte-identical and in place.

First construction of a store, its key and its genesis checkpoint is one transaction under the store's own lock, and the lock is taken before anything is created. Two consequences follow. Every file the attempt created is removed if any later step fails, including a failure at the lock itself, inside the creation of any one file, or at the checkpoint and its temporary file, so no partial store set is left behind; files that were already present are never removed. And two first constructors cannot divide the set between them, which would leave each believing the other's file was pre-existing and neither writing the genesis checkpoint.

The obligation to remove a file begins when the file does. Each name is created exclusively, and the identity used to remove it is recorded from that creation before any further step runs, so there is no interval in which a created file exists and nothing is responsible for it. Removal is bound to that identity, so a name replaced in the meantime is not removed.

A rename moves an inode between two names, and the obligation cannot be handed from the operation that created the file to the surrounding transaction in a single step. For that interval the creating operation is responsible for both names, and removes whichever holds the inode it created. Both removals are identity-bound, so a failed rename, or a final name that now holds a different file, removes nothing.

Which of the two names is covered depends on the caller, because a checkpoint replacement means different things to the two operations that perform one. First construction is building the store set in one transaction, so a checkpoint installed by a construction that then fails is removed with the rest of the set. An append writes and syncs its record before it replaces the checkpoint, so once that replacement commits the installed checkpoint authenticates a record that is already on disk, and the checkpoint it displaced authenticates a shorter log that no longer describes the store. An append therefore keeps the installed checkpoint when a later step fails, and reports the failure to the caller. Before the replacement commits, both callers behave alike: the temporary file is removed and an established checkpoint is left byte-identical.

Reads that decide a length read to the end of the file. This covers the HMAC key as well as the checkpoint: a key is accepted only when reading to the end yields exactly the key length, so neither a longer file whose first read returns the expected number of bytes nor a valid key delivered in small reads is misjudged.

One narrow exception is stated rather than left implicit. If the exclusive create succeeds but the immediately following inspection of its descriptor fails, no identity exists and the file is removed by name. That case rests on the cooperating-process boundary below.

It is the only removal of a store, key or checkpoint that is not identity-bound. It is not the only removal by name in the product: a failed initialisation removes its own staging directory, and a failed pointer write removes its own temporary file, each under a fresh unpredictable name it created. Those rest on the same boundary.

Temporary files carry the same rules. Each is created exclusively under a fresh unpredictable name, so a file already occupying a temporary name is never adopted, truncated, renamed or removed.

The rules above are the product's rules for a state file, not the receipt store's alone. The backend API key for an engine alias, the session record the supervisor writes for a running process, and the per-alias lock that serialises start, status and stop are held to them as well: a symbolic link at one of those names is refused rather than followed, a file that is not a regular file is refused without the product ever waiting on it, the mode is set through the descriptor the create returned, a read that decides a value reads to the end of the file within a declared bound, and a file the operation created is removed only while it is still that file. The directories holding them are covered too, and so is the path that reaches them. Every component of a declared state or sessions path is walked from the filesystem root through directory descriptors, and a symbolic link at any component is refused rather than followed. Checking the final name alone was not enough: the ancestors above it were resolved before that check ran, so a link at any of them put the whole state set on the far side of it while every check on the final name passed. Nothing above the declared path is trusted because it already exists.

A pathname is not authority over what it reaches. An environment variable, an operator's argument and the home anchor the operating system reports each select a spelling; none of them establishes that a link target substituted somewhere inside that spelling is one the product may write to. So no route converts a state path into its target before the walk, and the walk proves every component of whatever spelling it is given. A path the operator declares is used exactly as declared. The default path, used when the operator declares none, is assembled from spellings in the same way. A non-empty `XDG_STATE_HOME` is used without symbolic-link resolution, so a link at any component of it is refused by the walk rather than followed; the spelling is still normalised in the ordinary way, so a trailing separator or a `..` component is not preserved byte for byte. An empty `XDG_STATE_HOME` selects the home fallback, because taking an empty value as supplied would place state in a bare relative `plag-in` beneath the working directory. In the fallback, `.local`, `state` and `plag-in` are appended as literal components.

One exception is drawn as narrowly as the host requires, and it covers the home anchor alone. A host may reach the user's home through a compatibility link, and on such a host the product's own default would be refused by the product's own rule with nothing the operator could do about it. The anchor the operating system reports as the user's home is therefore translated, and nothing below it is. The consequence on a host where `/home` is a symbolic link to `/var/home` is that the product's own default is the path beneath `/var/home`, while a path an operator declares through `/home`, and any link at `.local`, at `state` or inside `XDG_STATE_HOME`, is refused with the component named. Directories the product must create above the declared one keep the process umask, since tightening a directory the operator owns is not this product's decision; the declared directory itself is confined to its owner. A key or record already present is read, never rewritten, and never removed.

A checkpoint already present beside a store and key that were not is refused, and the store and key this attempt created are removed. Such a checkpoint was authenticated under some other key, so the set it would complete cannot verify, and returning a larger invalid set than the one found is not a successful construction.

Two limits apply to all of the above. Setting the mode through the opened descriptor binds that one operation to the file that was opened; it is not a lasting confinement, because later reads and appends address the store by pathname. And the whole of this section holds against cooperating processes: a party that replaces one of these names between a check and its use is outside the threat model, as stated above.

The store, its HMAC key and its checkpoint are selected together from one reading of the pointer. They are never resolved by separate reads, which would allow an activation to land between two of them and pair one store with another store's key.

A gateway that is already serving follows the pointer. A store activated while it runs receives the next receipt it writes, with no restart and no requirement to stop the gateway before initialising. An append that has already begun completes against the store it resolved. Receipts written to the earlier store before activation stay in that store: they are reported by `plag-in status` under its path, and they are no longer served by request ID through the gateway, because the product reads the active store rather than falling back.

What the product will write as a pointer is what it will accept when reading one, for a store set that does not change between the two. A pointer naming a store that does not exist is refused at the write, not written and then refused at the next read. The bound is temporal and is stated rather than claimed away: the existence check and the commit are two operations, so a store removed between them leaves a committed pointer that the next read refuses. That window sits outside the cooperating-process boundary above, in the same way as any other replacement of one of these names between a check and its use.

The guarantee is stated against cooperating PLAG IN processes. Two locks carry it, and they cover different operations. Initialisation and rollback of a current-schema store take the transaction lock. Construction of a store, whether the original store or a current-schema one, takes that store's own lock. A process that takes neither and replaces one of these pathnames between a check and its use is outside the threat model.

Two receipt checks exist and report different things:

```text
plag-in receipt --check-chain
plag-in receipt --check-conformance
```

`--check-chain` reports storage integrity: the HMAC chain, the authenticated checkpoint and canonical storage, under the scope label `chain_checkpoint_storage`. It is not a conformance result and does not report a field named `valid`. `--check-conformance` runs every mandatory rule of the Inference Receipt Specification against the current store and reports the conformance level assessed, the errors and the checks it left unassessed.

### 2.4 Local model admission

Three axes are kept apart, and a refusal always names which one refused (D-021).

**Availability** is whether the complete model bytes are held on this computer: `local_complete`, `local_incomplete` or `remote_only`. Names and sizes are read from manifests and file lengths, so listing the inventory hashes no weight file. A manifest declaring no local model layer is `remote_only`. A declared blob that is absent, escapes the store, or whose length disagrees with its manifest is `local_incomplete`. So is a manifest whose declared digest is not exactly `sha256:` followed by 64 lowercase hexadecimal characters: such a manifest cannot bind a complete identity even when a file matching its malformed name exists. One validator applies that syntax across the full inventory, the quick summary and manifest-bound path resolution. The three surfaces refuse it differently and deliberately. The full held-model inventory keeps the entry visible as `local_incomplete` with reason `manifest_digest_malformed`, because that inventory is what an operator reads to learn why an entry cannot be started. The quick summary and manifest-bound resolution return no entry at all, as they do for any other unusable manifest layer, because neither is a completeness report. Absent bytes are never reported as an unsupported model, and no entry is dropped from the inventory for being unstartable: it stays visible with the reason.

**Compatibility** is `tested`, `unverified` or `unsupported`. `tested` requires an exact reviewed profile bound to an exact model identity, and nothing else confers it: not discovering a manifest, not parsing a GGUF file, not completing a trial, and not answering a request. `unsupported` requires a named structural or architectural reason read from the complete local bytes, such as a file that is not GGUF, a GGUF version this reader does not parse, or a file declaring no chat template. A model the host cannot currently fit is not `unsupported`.

The authority for `tested` is a reviewed compatibility record in the source, not a status field in a configuration file. Each record binds the complete SHA-256 of the exact model bytes, the identifier of the exact registered native ABI profile that loaded them, and the decision and completed trial that admitted the pair. A configuration profile may reference a record by identifier and may not supply one, so adding a record is a reviewed source change with its own regression coverage.

Configuration parsing refuses a `tested` profile whose record is absent or unregistered, refuses one whose engine is not the recorded native bundle, and refuses a `compatibility_record` on a profile that is not `tested`.

The stored label then selects a candidate and decides nothing else. Every route that resolves a model derives the compatibility state from the bytes it resolved, and carries that derived state into everything it reports, registers or records: the setup menu, the private test chat, the harness gateway, the embedded backend registration that feeds status, capabilities and receipts, configured-profile inspection and the direct serve command.

The derivation returns a compatibility evidence record of one shape on every outcome, carrying the state, the reason, the reviewed record identifier, the recorded model digest and the observed model digest. That record travels with the registered backend and appears on each surface that reports the state: the operator output of the private test chat and the harness gateway, configured-profile inspection, the status document, the capability document and the receipt adapter record. A surface never has to infer whether a digest is absent because it matched or because no record exists, and no surface accepts those values as a caller assertion. The registration that binds a backend derives its own state from the identity it is about to register and has no parameter through which a state could be supplied to it, so a state that conflicts with the resolved identity cannot enter from inside the product any more than from configuration. Bytes that are not the recorded bytes are reported `unverified` with both the recorded and the observed digest. A profile whose record is later withdrawn from the source is reported `unverified`, so a stored label never outlives its evidence. Because the menu offers a configured profile as a tested start, it resolves and hashes that profile's bytes to make the offer, which is the one place the inventory reads a weight file. Whether such bytes remain startable as a bounded unverified trial is a separate admission decision; they are never presented or recorded as tested. A report that resolves no model bytes, such as `doctor`, states which profiles are configured as tested and makes no compatibility finding.

**Admission** is a tested start, an approved unverified trial, or a refusal with a typed reason.

Selecting a model resolves the exact local bytes through the existing containment policy, refusing relative escape, absolute escape, symbolic-link escape, a missing blob and a digest that disagrees with its manifest. The complete selected bytes are hashed before any native load. A bounded reader then inspects only the GGUF header, metadata block and tensor descriptors, stopping at the aligned tensor-data offset: it never reads a tensor payload, never retains an array's elements, and validates every count, length and offset against an explicit limit and the file size before any seek or allocation. Truncation, an impossible offset, an unknown value type, an out-of-range integer and a limit breach each fail closed with a stable reason. Only values actually read are reported. A missing or unrecognised value stays `unassessed`, and no quantisation label is derived from the identifier that was read, because no locally held table binds the two for the build that wrote the file.

The existing cgroup, RAM and VRAM preflight then runs, unchanged and still labelled provisional. It is never weakened to admit a larger model. A refusal is a completed outcome carrying its measured inputs and thresholds, so a matrix records why an entry was not started rather than leaving it untried. A measurement that could not be taken is reported as unavailable, never as headroom.

An unverified trial requires a distinct confirmation after a disclosure of the exact tag, complete SHA-256, held size, parsed metadata, every unassessed field, the requested runtime settings, and the measured resources and thresholds, together with the statement that the trial promotes nothing. It uses an ephemeral profile for the life of one process and writes no configuration. Between that confirmation and the load, the file identity is re-checked, and the complete byte digest is re-verified again by the embedded engine before any native call. A load that fails to initialise the model, find a supported architecture, obtain a usable chat template or create the bounded runtime stops before listener readiness, releases every acquired handle, and writes no completed inference receipt.

Only the existing registered exact libllama ABI bundle is used. General model trial is not a route to arbitrary native runtimes, user-supplied shared libraries or remote fallback.

Status, capabilities and receipts report the compatibility state read from the confirmed profile at registration. A successful request never raises it. Receipts carry the state and the effective model identity inside PLAG IN's own namespaced adapter record, so the receipt schema is unchanged.

The private test chat may give the model an identity instruction built only from values verified on this computer: the local alias, the complete digest and the GGUF metadata that was read. It names no provider, is scoped to that conversation, and is never added to traffic from another harness. What the model answers about itself is its own claim and is never identity evidence, which the interface states to the operator.

## 3. Locality enforcement levels

The interface must never reduce locality to a Boolean.

| Level | Meaning | Permitted claim |
|---|---|---|
| L0 | Configuration inspected only | `locality configured, not enforced` |
| L1 | Gateway binds to loopback and contains no configured external inference route | `local route observed` |
| L2 | Operating-system network policy permits only required local communication | `local-only policy enforced for the measured process` |
| L3 | L2 plus an independent traffic observation over the test interval | `no external traffic observed during the stated interval` |

No level establishes what an unmeasured process did outside the recorded interval.

## 4. MVP architecture

### 4.1 Components

1. **Inventory**: read-only discovery of engines, manifests and model files.
2. **Registry**: stable aliases bound to exact model, engine and configuration digests.
3. **Embedded inference adapter**: loads a pinned `libllama` build and an exact local GGUF inside the PLAG IN process.
4. **Runtime manager**: owns model and context lifetime, serialises the initial single-user route, tracks readiness and refuses conflicting state.
5. **Gateway**: OpenAI-compatible chat, completion, embedding and model-list endpoints only where the selected engine supports them.
6. **Policy**: offline inference, loopback default, explicit private-network serving mode, authentication and bind controls.
7. **Receipt store**: append-only structured metadata, with content logging off by default.
8. **Operator CLI**: inspect, recommend, serve, stop, status, verify, receipt and connection export.
9. **Client interface**: a stable local HTTP boundary plus a PLAG IN capability document. Harness logic remains outside the gateway.
10. **Resource safety**: a verified systemd user scope, immediate RAM and memory-pressure admission, GPU-headroom measurement and bounded buffered setup interfaces.

### 4.2 MVP implementation

The private MVP uses Python 3.14 and the standard library where practical because the current workstation has Python 3.14.7 and no Go or Rust toolchain. The operator entry point remains `plag-in`. A later distribution assessment may replace the prototype with a compiled single binary. No single-binary distribution claim applies to the MVP.

The setup menu's private test chat is a bounded pre-release usability and connection check. It is not a conversation product, stored chat service or harness. The harness path starts the same bounded local gateway and exposes transport, protocol, base URL, scoped key and model alias. Harness configuration and workflow remain outside PLAG IN.

### 4.3 Existing model compatibility

The inventory may read Ollama manifests and blobs without changing them. It must not depend on the Ollama daemon for inference. The registry must also support directly named GGUF files under explicitly configured roots.

The target adapter loads the selected GGUF through pinned `libllama` functions inside PLAG IN. An Ollama-held blob, when selected, is treated as an exact local model file after manifest and digest validation. No Ollama API, Modelfile, daemon process or service setting enters the inference route.

The target architecture has one client-facing PLAG IN listener and no private backend listener. Clients cannot reach the embedded inference adapter except through PLAG IN authentication, policy and receipt handling.

The MVP chat route remains text-only. Structured image, audio and video content is refused before native inference. The embedded adapter does not implement URL resolution, remote model loading or file retrieval from request content. Plain URL text inside an ordinary string remains text.

The current Python implementation still contains a direct `llama-server` worker adapter. It is a transitional reference implementation and baseline. It does not satisfy the embedded target until the adapter, native dependency and failure-path tests in `EMBEDDED_LIBLLAMA_PLAN.md` pass.

High-throughput inference systems are not part of the MVP route. A project called a server may still run locally, but local execution does not establish offline operation, absence of telemetry or a complete dependency identity. D-012 permits a later runtime only after its complete offline package, dependencies, network behaviour and effective settings pass the same identity, locality and receipt contract. PLAG IN does not route private inference to a supplier-hosted endpoint.

### 4.4 Runtime profile

The private MVP uses one explicit starting profile:

| Setting | Initial value | Status |
|---|---:|---|
| Context size | 8192 tokens | unmeasured starting value |
| GPU layers | `auto` | requested starting policy, effective value must be measured |
| CPU threads | `-1` | automatic selection, recorded explicitly |
| Batch size | 2048 | unmeasured starting value |
| Micro-batch size | 512 | unmeasured starting value |
| Parallel slots | 1 | bounded single-user starting value |
| Temperature | 0.8 | server fallback only |
| Top-p | 0.95 | server fallback only |

The operator sees and confirms the exact model path, weight digest, size and requested runtime profile before loading weights. A non-interactive caller must supply `--confirm-runtime-profile` after reviewing the same values. Status and receipts distinguish requested settings from effective settings queried after model and context creation. Quantisation is fixed by the selected model bytes. Its human-readable label remains `unverified` until a deterministic GGUF metadata reader supplies it.

### 4.5 API boundary

MVP endpoints:

- `GET /health`;
- `GET /v1/models`;
- `POST /v1/chat/completions`;
- `POST /v1/completions`, only when supported;
- `POST /v1/embeddings`, only when supported;
- `GET /plag-in/v1/status`;
- `GET /plag-in/v1/receipts/{request_id}`.

Unsupported capabilities return a typed error. They must not be simulated by changing the request silently.

`CLIENT_INTERFACE.md` controls UI and harness integration. It requires `/v1/chat/completions` and `/v1/models`. Responses, embeddings, completions and Anthropic Messages are capability-gated. PLAG IN never executes tools or owns an agent loop.

## 5. Model recommendation policy

Recommendations must come from a versioned catalogue with source records. The selection function is deterministic for the same catalogue and answers.

The catalogue may include model metadata and measured local results. A model-generated preference is not a verification decision.

The gateway must state separately:

- what is compatible;
- what fits estimated hardware limits;
- what passed a local task test;
- what is merely recommended for trial.

## 6. Deliberate exclusions from MVP

- chat UI;
- agent loops;
- MCP tool execution;
- document retrieval;
- prompt marketplace;
- automatic cloud fallback;
- automatic model downloads;
- cluster scheduling;
- billing;
- user analytics;
- stored conversation history;
- harness-specific workflow code;
- MCP tool execution;
- Windows and macOS support;
- marketing claims about privacy, speed or market demand;
- any claim of universal model support: the truthful claim is that PLAG IN can attempt one bounded trial of a complete local GGUF through its registered libllama runtime and report the observed result;
- a preferred vendor, model or harness: no route reads a client's name, user-agent or executable, no model is ranked by vendor or name, and no protocol is added because the model that implemented it belongs to that provider.

## 7. Success condition

The MVP proceeds beyond private trial only if it demonstrates all of the following:

1. A held model is served through no more than three user commands after the binary is present.
2. A missing model produces no more than three traceable options and no implicit network action.
3. The locality report distinguishes configuration, enforcement and observation.
4. A receipt associates each successful request identifier with an exact weight digest, and with the declared runtime state where the emitting adapter reports one, under a local HMAC. The runtime blocks are optional in the portable specification, and a receipt carrying none of them is still conforming at its level. It does not bind response content.
5. The deterministic acceptance suite passes.
6. The comparison against direct `llama-server` and llama-swap identifies a material operational advantage.

If item 6 fails, retain the work as an internal integration profile rather than a separate product.
