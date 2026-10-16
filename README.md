# PLAG IN

**Local inference controls with inspectable metadata receipts.**

PLAG IN is an experimental source gateway for direct, inspectable access to local GGUF models. The statement above is the product requirement. It is not a blanket privacy guarantee.

Version `0.1.0a3` supports Linux and Python 3.14 or newer. It does not download models, install a native runtime, execute tools or silently fall back to a remote provider.

## Quickstart: try it without a model

You need Linux and Python 3.14 or newer with `venv` and `pip` support. Start in the root of a fresh source checkout, the directory containing `pyproject.toml`. No model or native runtime is needed for installation, the doctor report or the synthetic tests.

**For inference later:** you must already hold a GGUF model and a native `libllama` bundle whose complete component digests match a [registered native ABI profile](src/plag_in/native_profiles.py). An arbitrary `libllama` build is not sufficient. Supporting another build requires a reviewed profile and regression coverage; changing a filename or configuration label cannot admit it.

Create a virtual environment. Run in your terminal:

```text
python3 -m venv .venv
```

Install from this checkout. Run in your terminal:

```text
.venv/bin/python -m pip install .
```

This source installation may download the declared build dependency, `setuptools>=68`. It does not install a model or native runtime. For a machine without network access, see [Offline installation](#offline-installation).

Check the installed command. Run in your terminal:

```text
.venv/bin/plag-in doctor --format json
```

The command should exit successfully and print JSON. With no configuration supplied, `configuration.status` is `not_supplied`, and `locality_controls.status` and `runtime_settings.status` are `unconfigured`. These are expected results, not an installation failure. The report records `model_started: false`; it does not establish native-runtime availability, model compatibility or inference readiness.

You can now run the [synthetic tests](#verify) without a model. If you already have the required native bundle and model, continue to [Interactive setup](#interactive-setup).

## Historical evidence (0.1.0a2)

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

## Offline installation

Use this alternative only when the build toolchain, including `setuptools>=68` and wheel-building support, is already installed for your chosen Python 3.14 or newer interpreter. It reuses that interpreter's packages for the build; the environment is not isolated from those packages. Start in a fresh source checkout without an existing `.venv`.

Run in your terminal:

```text
python3 -m venv --system-site-packages .venv
```

Confirm the available setuptools version is at least 68. Run in your terminal:

```text
.venv/bin/python -c "import setuptools; print(setuptools.__version__)"
```

Install using that existing build backend, without consulting a package index. Run in your terminal:

```text
.venv/bin/python -m pip install --no-index --no-build-isolation .
```

Then run the same doctor check from the quickstart. If the build dependency is unavailable or too old, supply it through your normal package-management process before trying this route. Do not lower the requirement in `pyproject.toml`.

## Interactive setup

After installation, open the menu in an interactive terminal. Run in your terminal:

```text
.venv/bin/plag-in
```

The menu uses the arrow keys and Enter; choose **Exit** or press `q` to close it. Piped or redirected input is not an interactive terminal, and the no-argument command returns usage instead of opening setup.

Choose **Inspect this computer** to view the doctor report. **Choose or configure a local model** recognises the exact registered Qwen2.5 3B model and native bundle at the locations checked by [the setup flow](src/plag_in/setup_flow.py). It is not a general model or runtime installer. If those files do not match, no configuration is written; another GGUF may still be eligible for an explicitly configured unverified trial under [Model admission](#model-admission).

Configuration replacement, receipt-store creation and native model loading each require confirmation. Review the proposed configuration and requested runtime settings before starting a private test chat. The install check alone is not a reason to approve a load.

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

The current alpha uses a generated local API key. Per-client identities and least-privilege key scopes are not implemented in `0.1.0a3`. Browser-origin and Host-header hardening are also not claimed.

## Verify

From the source checkout root, run the full synthetic suite. Installation is not required: `PYTHONPATH=src` makes the source-layout package importable. Run in your terminal:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -W error::ResourceWarning -m unittest discover -s tests -t .
```

The suite uses fixture engines and native doubles. Some tests bind loopback ports. A sandbox that denies socket creation cannot run those tests; report that restriction separately, without skipping tests or treating a partial run as a full pass. Plain discovery from an uninstalled checkout without `PYTHONPATH=src` cannot import the package.

View the standalone receipt verifier's arguments. Run in your terminal:

```text
python3 tools/verify_inference_receipts.py --help
```

See [product specification](PRODUCT_SPEC.md), [client interface](CLIENT_INTERFACE.md), [verification boundary](docs/VERIFICATION.md), [security policy](SECURITY.md) and [release notes](RELEASE_NOTES.md).

## Licence

Licensed under the Apache License 2.0. See [LICENSE](LICENSE).
