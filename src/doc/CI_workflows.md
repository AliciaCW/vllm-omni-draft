## CI Workflows Overview

This PR introduces three GitHub Actions workflows to standardize CI for the repository. These files are currently placed under `src/CI_test/.github/workflows/` and can be moved to the repository root at `.github/workflows/` when ready to enable them.

### ci.yml
- Purpose: Run test suite across multiple Python versions.
- Triggers: `push` (branch: dev_CI), `pull_request` (into main), and manual `workflow_dispatch`.
- Key steps:
  - Checkout repository.
  - Set up Python (matrix: 3.9, 3.10, 3.11) with pip cache.
  - Install dependencies.
  - Run `pytest` with verbose output, excluding tests marked `slow`.
- Stability:
  - `timeout-minutes: 30` to avoid hanging jobs.
  - `strategy.fail-fast: false` to allow all matrix runs to complete.

### pre-commit.yml
- Purpose: Enforce code style and basic static checks using pre-commit hooks.
- Triggers: `pull_request` and `push` to `main`.
- Key steps:
  - Checkout repository.
  - Set up Python 3.11 with pip cache.
  - Run `pre-commit` across all files with `--hook-stage manual`.
- Note: Requires a valid `.pre-commit-config.yaml` in the repository.

### automerge.yml
- Purpose: Automatically merge eligible pull requests.
- Triggers: PR events, reviews, and completed check suites.
- Permissions: `contents: write`, `pull-requests: write`.
- Policy (environment variables):
  - `REQUIRED_LABELS: automerge` — only PRs with this label are merged.
  - `REQUIRED_STATUS_CHECKS: true` — merges only after all checks pass.
  - `MERGE_METHOD: SQUASH` — unify commits on merge.
  - `UPDATE_METHOD: rebase` — keep PR branch up to date before merge.

### Notes
- These workflows are minimal and safe by default.
- They can be extended later (e.g., caching, additional test jobs, artifact uploads) as CI needs evolve.

