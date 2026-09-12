# Contributing

1. Create a feature branch.
2. Keep deterministic geoscience calculations independent from provider-specific AI code.
3. Add tests for new numerical behavior.
4. Run:
   ```bash
   ruff check .
   pytest
   ```
5. Update documentation for user-facing changes.
6. Never commit credentials or proprietary datasets.

## Commit style

Prefer concise commits such as:

- `feat: add LAS parser`
- `feat: add blind-well validation`
- `fix: handle missing RHOB`
- `docs: update petrophysics assumptions`
