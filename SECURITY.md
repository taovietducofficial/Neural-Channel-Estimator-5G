# Security Policy

This is a portfolio/research project (see `README.md`), not a production
system with live user data — but if you find a genuine security issue
(e.g. a vulnerability in `service/app.py`'s handling of untrusted input, or
a dependency with a known CVE not yet caught by Dependabot), please report
it privately rather than opening a public issue.

## Reporting

Use GitHub's private vulnerability reporting for this repository
(Security tab → "Report a vulnerability"), or open a regular issue if the
concern is not security-sensitive (e.g. a bug, not an exploit).

## Scope

- `service/` — the FastAPI inference service and its input validation.
- Dependency vulnerabilities flagged by Dependabot (`.github/dependabot.yml`).

Out of scope: this repository does not run as a live, publicly reachable
deployment anywhere; there is no production instance to attack.
