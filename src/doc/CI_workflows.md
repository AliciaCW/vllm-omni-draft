## CI Workflows Overview

This document describes the GitHub Actions workflows in the vllm-omni repository.

### ci.yml

- **Purpose**: Smart test workflow that prioritizes testing changed files, with fallback to testing all files.
- **Triggers**: 
  - `push` to **all branches** (`branches: ["**"]`)
  - `pull_request` (into main)
  - Manual `workflow_dispatch` with optional inputs:
    - `test_changed_only`: Only test changed files (default: `'true'`)
    - `enable_changed_files_detection`: Enable changed files detection (default: `'true'`)
  - `workflow_call`: Can be invoked by other workflows
- **Key steps**:
  - Checkout repository.
  - Free disk space (remove unnecessary toolchains, prune docker).
  - Set up Python (matrix: 3.12, 3.13) with pip cache.
  - Install dependencies with `--no-cache-dir` to reduce disk usage.
  - Determine test scope based on inputs or defaults.
  - Detect changed test files (if enabled and needed).
  - Run tests:
    - **Priority**: Test only changed files (if changes detected and corresponding tests found).
    - **Fallback**: Test all files (if detection fails, no tests found, or detection disabled).
- **Test strategy**:
  - Excludes tests marked `slow`.
  - Runs `pytest tests/ -v -m "not slow"`.
  - Timeout: 30 minutes.
  - Matrix testing: Continues running other versions even if one fails.
- **Fallback mechanism**:
  - If `test_changed_only` is `'false'` → test all files.
  - If `enable_changed_files_detection` is `'false'` → test all files.
  - If no corresponding test files found → test all files.

### pre-commit.yml

- **Purpose**: Enforce code style and static checks using pre-commit hooks on changed files only.
- **Triggers**: 
  - `pull_request` (all PRs)
  - `push` to all branches (`branches: ["**"]`)
  - Manual `workflow_dispatch`
- **Key steps**:
  - Checkout repository.
  - Set up Python 3.12 with pip cache.
  - Run `pre-commit` with `--hook-stage manual` (checks changed files only, not all files).
  - Executes all hooks defined in `.pre-commit-config.yaml`.
- **Checks performed**:
  - YAML syntax (`check-yaml`)
  - Debug statements (`debug-statements`)
  - File formatting (`end-of-file-fixer`, `mixed-line-ending`, `trailing-whitespace`)
  - Python code formatting (`black`, `isort`)
  - Python linting (`ruff`)
  - Spell checking (`typos`)
  - GitHub Actions workflow validation (`actionlint`)
- **Concurrency**:
  - Uses concurrency groups to prevent duplicate runs.
  - Automatically cancels in-progress runs for pull requests.
- **Note**: Requires a valid `.pre-commit-config.yaml` configuration file.

### ci-all-files.yml

- **Purpose**: Run the full test suite, testing all files (no change detection).
- **Triggers**: 
  - Manual `workflow_dispatch` only
- **Key steps**:
  - Checkout repository with `fetch-depth: 0` (full history needed).
  - Free disk space.
  - Set up Python (matrix: 3.12, 3.13) with pip cache.
  - Install dependencies.
  - Run all tests: `pytest tests/ -v -m "not slow"`.
- **Use cases**:
  - Verify the entire codebase test status.
  - Pre-release full test suite.
  - Periodic comprehensive testing.
- **Difference from ci.yml**:
  - `ci.yml`: Smart detection, prioritizes changed files.
  - `ci-all-files.yml`: Always tests all files, no change detection.

### automerge.yml

- **Purpose**: Automatically merge eligible pull requests.
- **Triggers**: 
  - `pull_request` events:
    - `labeled`: PR labeled
    - `synchronize`: PR updated with new commits
    - `opened`: PR opened
    - `ready_for_review`: PR ready for review
    - `reopened`: PR reopened
  - `pull_request_review`: New review submitted
  - `check_suite`: CI checks completed
- **Permissions**: 
  - `contents: write` — required to merge PRs
  - `pull-requests: write` — required to update PR status
- **Merge policy**:
  - `MERGE_METHOD: SQUASH` — unify commits on merge.
  - `UPDATE_METHOD: rebase` — keep PR branch up to date.
  - `REQUIRED_LABELS: automerge` — only PRs with this label are merged.
  - `REQUIRED_STATUS_CHECKS: true` — merges only after all checks pass.
- **Safety**:
  - Only merges PRs explicitly labeled with `automerge`.
  - Requires all CI checks to pass.
  - Uses squash merge to keep history clean.

## Workflow Comparison

| Workflow             | Test Scope                     | Triggers                             | Purpose              |
| -------------------- | ------------------------------ | ------------------------------------ | -------------------- |
| **ci.yml**           | Changed files, fallback to all | All push, PR to main, manual trigger | Daily CI, smart test |
| **pre-commit.yml**   | Changed files only             | All PR, all push, manual trigger     | Code style checks    |
| **ci-all-files.yml** | All files                      | Manual trigger only                  | Full test validation |
| **automerge.yml**    | No tests, merge only           | PR-related events                    | Auto-merge PR        |

## Usage Guidelines

### Daily Development
- **Committing code**: Local pre-commit hooks automatically check staged files.
- **Pushing code**: `ci.yml` and `pre-commit.yml` run automatically.
- **Creating PR**: Both workflows run automatically.

### Manual Trigger
- **Quick validation**: Use `ci.yml` with changed files only option.
- **Full test suite**: Use `ci-all-files.yml` to test all files.
- **Code checks**: Use `pre-commit.yml` for code style validation.

### Auto-merge
- Add `automerge` label to PR.
- Ensure all CI checks pass.
- Workflow will automatically merge the PR.

## Notes

- These workflows are minimal and safe by default.
- They can be extended later (e.g., caching, additional test jobs, artifact uploads) as CI needs evolve.
- **Test coverage**: `ci.yml` has a fallback mechanism to ensure no tests are missed.
- **Code quality**: `pre-commit.yml` checks only changed files for efficiency.
- **Security**: `automerge.yml` requires explicit labels and all checks to pass.
- **Performance**: Changed file detection significantly reduces CI runtime.
- **Extensibility**: All workflows can be extended as needed.

