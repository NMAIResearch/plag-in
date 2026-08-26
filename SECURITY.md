# Security policy

PLAG IN 0.1.0a1 is an experimental source alpha. Do not use it for sensitive or production workloads.

## Supported scope

Security fixes are considered only for the current alpha source. Linux is the only tested platform. No stability or response-time commitment is offered.

## Reporting

Send non-public vulnerability reports to `NMAIResearch@proton.me`. Include the affected version, operating system, reproduction steps and the security boundary you expected.

Do not include model weights, private prompts, credentials or unrelated personal information.

## Boundary

- Loopback binding and authentication are application controls.
- Content-free receipts do not prove that another process retained no content.
- Embedded inference does not prove absence of network traffic.
- The application does not install a host firewall or general operating-system sandbox.
- Native libraries execute inside the PLAG IN process and can terminate or corrupt it.
- A non-loopback deployment requires separate network, authentication and threat review.
