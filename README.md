# PLAG IN

PLAG IN is a source alpha for running an exact local GGUF model through a small authenticated gateway while recording content-free inference receipts.

The alpha is intentionally narrow. It supports Linux and Python 3.14 or newer. Its confirmed setup route recognises one exact Ollama v0.32.13 native bundle and one already-held Qwen2.5 3B model. It does not download models, install a native runtime or silently fall back to a remote provider.

## Status

The fixture suite passed 311 of 311 tests, N=311, on 2026-08-26. One bounded live L3 trial completed one synthetic request and wrote one valid schema-v3 receipt, N=1 request.

This does not establish production readiness, repeated-run reliability, general model quality or comparative performance. No direct `llama-server` or llama-swap performance baseline has been run.

Ordinary alpha operation does not itself install an operating-system egress policy. The accepted L3 trial used a separate Bubblewrap and seccomp execution envelope. Configuration alone is not proof of locality.

## What it provides

- Exact model-file hashing and containment checks.
- Registered native ABI profiles bound to complete component digests.
- Embedded `libllama` inference with no child inference process.
- An authenticated local Chat Completions v1 subset.
- Requested and effective runtime-profile records.
- Content-free, hash-chained inference receipts.
- Bounded memory and swap policy for shipped model-loading commands.
- A read-only `doctor` inspection and confirmed first-run workflow.

PLAG IN is not a model distributor, chat application, agent framework or substitute for operating-system isolation.

## Install from source

Run in your terminal:

```text
python3 -m venv .venv
```

Then run in your terminal:

```text
.venv/bin/python -m pip install .
```

Inspect the host without loading a model:

```text
.venv/bin/plag-in doctor --format json
```

Open the confirmed setup menu:

```text
.venv/bin/plag-in
```

The editable or source installation may obtain the declared build dependency if it is not already installed. It never installs a model.

## Tested-profile limitation

The confirmed setup route matches exact file digests. A library with the same name or version label but different bytes is refused. Other models and native builds require a separately reviewed ABI profile and compatibility test.

The alpha does not ship Ollama, `libllama`, CUDA libraries, model weights or an installer. A host without the exact registered bundle can use `doctor`, inspect the source and run the fixture suite, but cannot use the confirmed model-serving path.

## Verify

Run in your terminal:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -W error::ResourceWarning -m unittest discover -s tests -t .
```

Some tests bind loopback ports. A sandbox that denies socket creation cannot run those tests.

See [verification boundary](docs/VERIFICATION.md), [security policy](SECURITY.md) and [release notes](RELEASE_NOTES.md).

## Licence

Licensed under the Apache License 2.0. See [LICENSE](LICENSE).
