# Verification boundary

Purpose: state what has and has not been established for PLAG IN 0.1.0a1.

## Established in the fixture suite

The declared suite passed 311 of 311 tests, N=311, on 2026-08-26. It covers configuration validation, identity binding, path containment, receipt integrity, API authentication and scope, setup controls, resource-policy paths, embedded-adapter fixtures and direct-worker regressions.

Fixture engines and native doubles do not establish live model behaviour.

## Established in one live trial

One synthetic request completed through the embedded route and produced one valid schema-v3 receipt, N=1 request. The run used exact registered native and model digests inside a network-denied, read-only execution envelope.

This is one accepted bounded result. It is not a repeated-run or performance result.

## Unassessed

- Repeated-run reliability.
- General output quality.
- Direct `llama-server` and llama-swap comparative performance.
- Exact GPU layer offload.
- Windows and macOS behaviour.
- Python versions below 3.14.
- Other models, native builds and hardware combinations.
- Public-network or multi-user deployment.

## Receipt level

The persistent receipt attests the application-observable L1 state and exact runtime identities. The L3 conclusion depends on the separate operating-system trial procedure and its reviewed execution evidence. A receipt alone does not prove L3 locality.
