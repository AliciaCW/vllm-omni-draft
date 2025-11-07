## CI Workflows Overview

This PR introduces three GitHub Actions workflows to standardize CI for the repository. 

### ci.yml
- Purpose: Run the full test suite across multiple Python versions; default workflow for pushes and PRs.
- Triggers: `push` (branch: dev_CI), `pull_request` (into main), and manual `workflow_dispatch`.
- Key steps:
  - Checkout repository.
  - Set up Python (matrix: 3.9, 3.10, 3.11) with pip cache.
  - Install dependencies.
  - Run `pytest` with verbose output, excluding tests marked `slow`.
- Stability:
  - `timeout-minutes: 30` to avoid hanging jobs.
  - `strategy.fail-fast: false` to allow all matrix runs to complete.

### ci-changed-files.yml
- Purpose: Run targeted tests only for files changed in a PR or selected workflow run.
- Triggers: Manual `workflow_dispatch` or `workflow_call` from other workflows.
- Key steps:
  - Checkout repository with full history.
  - Set up Python (matrix: 3.9, 3.10, 3.11) with pip cache.
  - Determine whether to run selective tests or fall back to the full suite.
  - Invoke `.github/scripts/detect_changed_tests.sh` to map changed source files to associated tests.
  - Run `pytest` on the filtered list or entire `tests/` directory when needed.
- Controls:
  - Inputs `test_changed_only` and `enable_changed_files_detection` allow manual overrides.
  - Falls back to full test run if no matching tests are detected.

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

