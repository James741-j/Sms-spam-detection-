# Security Policy

## Supported version
Only the latest commit on `main` is supported.

## Reporting a vulnerability
Please do not open a public issue for security problems. Report them privately
using GitHub's "Report a vulnerability" option under the **Security** tab, or
email the repository owner. You will receive a reply within 7 days.

## Security measures in this project
- Bandit scans the source code on every push.
- pip-audit checks dependencies for known vulnerabilities.
- Gitleaks scans the repository for committed secrets.
- User input is validated (type, empty, maximum 1000 characters) before use.
- The Docker image runs as a non-root user.
