# Security Policy

## Reporting a vulnerability

**Do not open public issues for security reports.**

Two channels (use either):

1. **GitHub Security Advisory** — preferred. Open a draft advisory at
   <https://github.com/krishnendu/stratum/security/advisories/new>. This
   creates a private fork the maintainer can collaborate on.
2. **Email** — `security@stratum.dev` (alias pending domain registration).
   Until the alias is live, email the maintainer privately via GitHub
   (Settings → Notifications → "Email the owner").

## Response targets

- Acknowledgement within 72 hours.
- Mitigation plan or fix within 14 days for critical issues.
- Public CVE assignment via GitHub advisories where applicable.
- Embargo respected on request.

## Scope

In scope:

- The Stratum binary, its workspace crates, and shipped configuration
  defaults.
- Tool sandbox profiles in `plan/31-tool-sandbox-and-secrets.md` (private).
- Update / telemetry / crash-report pipelines once they land.

Out of scope:

- Third-party model weights (report upstream).
- Issues in pre-alpha proof-of-concept code; please file as a normal bug if
  you're unsure.

## Supported versions

Pre-1.0: only the latest tagged release. Once the project tags v1.0, the
support window will be documented here.
