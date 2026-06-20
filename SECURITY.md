# Security Policy

## Supported versions

The latest commit on the `main` branch is the supported version.

## Reporting a vulnerability

Please report security issues **privately**, not as a public issue.

- Preferred: open a private report via GitHub's
  [Security advisories](https://github.com/ramenprotokol/hallucination-hunter/security/advisories/new).
- You can expect an initial response within 7 days.

## Handling of secrets

`hallucination-hunter` reads provider API keys (e.g. `ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`) from environment variables only. It never logs, stores, or
transmits them anywhere other than the provider's official SDK. The default
detector and the test suite run fully offline with no keys at all. Do not commit
`.env` files — they are gitignored.
