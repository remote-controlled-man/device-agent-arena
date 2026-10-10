# Security policy

DeviceAgent Arena is a research benchmark and local post-training sandbox, not a
production device-control service. Do not connect its simulated tool interfaces to
real devices or privileged systems without adding independent authentication,
authorization, confirmation, sandboxing, rate limits, and operational monitoring.

## Reporting a vulnerability

For a public GitHub repository, use GitHub's private security advisory flow rather
than opening a public issue. Include the affected revision, a minimal reproduction,
the impact, and any suggested mitigation. Do not include real credentials, user
data, or an unpublished test-generation secret.

## Secrets and model output

- No API keys, Hugging Face tokens, hidden test secrets, or user data belong in the
  repository or its issue tracker.
- Model output is untrusted. The Harness parses it into typed actions and validates
  tool names, arguments, preconditions, permissions, and state transitions before
  execution.
- The curated public release intentionally excludes model weights and local caches.
  Obtain third-party models from their original distributors under their own terms.

Supported versions are the current `main` branch and the latest tagged release.
