# Contributing

## Development setup

1. Create a virtual environment outside version control.
2. Install the package with `python -m pip install -e ".[test]"`.
3. Run `python -m compileall -q harness real_ai_harness.py`.
4. Run `pytest -q tests/unit` and `pytest -q tests/integration`.

## Pull requests

- Keep each pull request focused and explain the user-facing impact.
- Add or update tests for behavior changes.
- Use concise, imperative commit subjects.
- Do not commit credentials, machine-local paths, runtime state, benchmark output,
  models, datasets, caches, virtual environments, native binaries, or archives.
- Keep source and documentation reviewable as plain text. Generated evidence belongs
  in GitHub Actions artifacts or an external, versioned data store.

## Security

Do not open a public issue for a suspected vulnerability. Follow
[`SECURITY.md`](SECURITY.md) instead.
