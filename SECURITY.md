# Security policy

PLAG IN `0.1.0a2` is an experimental source alpha. Do not use it for sensitive or production workloads.

## Supported scope

Security fixes are considered for the current alpha source. Linux is the only tested platform. No stability or response-time commitment is offered.

## Reporting

Send non-public vulnerability reports to `NMAIResearch@proton.me`. Include the affected version, operating system, reproduction steps and the security boundary expected.

Do not include model weights, private prompts, credentials or unrelated personal information.

## Implemented controls

- The gateway binds to loopback and requires an API key.
- Configuration, model loading and receipt-store creation require explicit operator actions.
- Remote fallback is absent.
- Model identity derives from complete local bytes, not from a tag or model response.
- Registered native profiles bind complete component digests.
- Bounded GGUF metadata inspection runs without retaining tensor data.
- Request bodies, generated tokens, memory, swap and concurrency are bounded.
- Receipts retain no prompt, response or digest derived from either.
- Unsupported protocol features are refused rather than ignored.
- Historical receipt stores are preserved rather than rewritten during schema migration.

## Boundary and unassessed risks

- Loopback binding and API-key authentication are application controls. They are not an operating-system firewall.
- The alpha does not claim complete Host-header, DNS-rebinding, browser-origin or CORS protection.
- One generated key authorises the local API. Per-client identities and least-privilege scopes are not implemented.
- A valid GGUF structure is not evidence that a model is benign. Untrusted bytes still reach native code if the operator confirms a load.
- Native libraries execute in the gateway process and can terminate or corrupt it.
- A registered native digest establishes exact identity, not absence of vulnerabilities.
- Embedded inference does not prove absence of network traffic. Ordinary operation installs no egress policy and performs no continuous traffic observation.
- Content-free receipts do not prove that another process retained no content.
- Local HMAC integrity is operator-verifiable, not third-party authenticity.
- A non-loopback or multi-user deployment requires a separate network, authentication and threat review.
- Prompt injection, unsafe generated content and false model self-identification remain outside the gateway's model-admission evidence.

## Disclosure handling

Reports should identify the exact source version, runtime source digest where available, affected route or state file and whether a model or native loader was involved. A report about an unsupported feature is not a vulnerability unless the gateway accepts it contrary to its declared boundary.
