# CI 测试范围控制说明

本文档说明如何控制 CI 测试的范围：是测试所有文件还是只测试变更的文件。

## 架构说明

变更文件检测功能已分离到独立的 workflow 文件中，保持 `ci.yml` 简洁。

- **ci.yml**: 基础 CI 工作流，负责运行所有测试（默认行为）
- **ci-changed-files.yml**: 变更文件检测工作流，负责检测变更文件并只测试相关测试
- **detect_changed_tests.sh**: 独立的检测脚本，负责检测变更文件和匹配测试文件

### 工作流选择

- **使用 `ci.yml`**: 运行所有测试（默认，适合 push 和重要合并）
- **使用 `ci-changed-files.yml`**: 只测试变更的文件（适合 PR 和快速验证）

## 默认行为

### Pull Request (PR)
- **默认**: 使用 `ci.yml`，测试所有文件
- **原因**: 确保 PR 不会破坏现有功能
- **可选**: 可以手动触发 `ci-changed-files.yml` 来只测试变更的文件（加快 CI 速度）

### Push 事件
- **默认**: 使用 `ci.yml`，测试所有文件
- **原因**: Push 到主分支需要确保所有测试通过，验证整体代码质量

### 手动触发 (workflow_dispatch)
- **ci.yml**: 测试所有文件（默认）
- **ci-changed-files.yml**: 可配置参数：
  - `test_changed_only`: 是否只测试变更文件（默认 `true`）
  - `enable_changed_files_detection`: 是否启用变更文件检测功能（默认 `true`）

## 如何控制测试范围

### 方法 1: 选择不同的 workflow

1. **测试所有文件**: 使用 `ci.yml` workflow
   - 访问 GitHub Actions 页面
   - 选择 "CI" workflow
   - 点击 "Run workflow"

2. **只测试变更文件**: 使用 `ci-changed-files.yml` workflow
   - 访问 GitHub Actions 页面
   - 选择 "CI Changed Files" workflow
   - 点击 "Run workflow"
   - 配置参数：
     - **Only test changed files**: `true` (只测试变更，默认) 或 `false` (测试所有)
     - **Enable changed files detection**: `true` (启用检测，默认) 或 `false` (禁用检测，强制测试所有)

### 方法 2: 在 ci.yml 中调用 ci-changed-files.yml（高级）

如果需要从 `ci.yml` 中调用变更文件检测功能，可以添加 workflow_call：

```yaml
jobs:
  test-changed:
    uses: ./.github/workflows/ci-changed-files.yml
    with:
      test_changed_only: 'true'
      enable_changed_files_detection: 'true'
```

### 方法 3: 禁用检测功能

如果不想使用变更文件检测功能：

1. **手动触发时**: 在 `ci-changed-files.yml` 中设置 `enable_changed_files_detection: false`
2. **直接使用 ci.yml**: 使用基础的 `ci.yml` workflow，它会测试所有文件

## 变更文件检测逻辑

检测逻辑位于 `.github/scripts/detect_changed_tests.sh` 脚本中。

### 检测的变更文件类型
- Python 源代码文件 (`*.py`)
- 测试文件 (`tests/**/*.py`)

### 测试文件匹配规则

当源代码文件变更时，CI 会尝试查找对应的测试文件：

1. **变更测试文件**: 直接运行该测试文件
2. **变更源代码文件**: 查找对应的测试文件，支持以下命名模式：
   - `tests/test_<module_path>.py` (例如: `vllm_omni/core/scheduler.py` → `tests/test_core_scheduler.py`)
   - `tests/<module_path>_test.py` (例如: `vllm_omni/core/scheduler.py` → `tests/core_scheduler_test.py`)
   - `tests/<dir>/test_<module>.py` (例如: `vllm_omni/core/scheduler.py` → `tests/core/test_scheduler.py`)
   - `tests/<dir>/<module>_test.py` (例如: `vllm_omni/core/scheduler.py` → `tests/core/scheduler_test.py`)

### 找不到对应测试文件时

如果变更的源代码文件没有找到对应的测试文件，CI 会：
- **回退到测试所有文件**，确保不会遗漏测试

## 示例场景

### 场景 1: PR 只修改了一个模块
```
变更文件: vllm_omni/core/scheduler.py
找到测试: tests/test_core_scheduler.py
运行: pytest tests/test_core_scheduler.py -v -m "not slow"
```

### 场景 2: PR 修改了多个文件
```
变更文件: 
  - vllm_omni/core/scheduler.py
  - vllm_omni/engine/processor.py
  - tests/test_core_scheduler.py
找到测试: 
  - tests/test_core_scheduler.py (直接匹配)
  - tests/test_engine_processor.py (自动匹配)
运行: pytest tests/test_core_scheduler.py tests/test_engine_processor.py -v -m "not slow"
```

### 场景 3: Push 到主分支
```
事件: push
默认行为: 测试所有文件
运行: pytest tests/ -v -m "not slow"
```

### 场景 4: 手动触发，选择只测试变更文件
```
Workflow: ci-changed-files.yml
事件: workflow_dispatch
参数: test_changed_only = true, enable_changed_files_detection = true
行为: 检测变更文件并只测试相关测试
```

### 场景 5: 手动触发，测试所有文件
```
Workflow: ci.yml
事件: workflow_dispatch
行为: 测试所有文件
```

### 场景 6: 禁用检测功能，强制测试所有文件
```
Workflow: ci-changed-files.yml
事件: workflow_dispatch
参数: enable_changed_files_detection = false
行为: 跳过检测步骤，直接测试所有文件
```

## 注意事项

1. **首次合入 CI**: 当首次添加 CI 配置时，建议先运行一次"测试所有文件"，确保所有现有代码的测试都能通过

2. **测试文件命名**: 为了确保 CI 能正确找到测试文件，建议遵循以下命名规范：
   - `tests/test_<module_name>.py`
   - `tests/<module_name>_test.py`

3. **慢测试标记**: 无论测试范围如何，都会跳过标记为 `slow` 的测试

4. **Git 历史**: CI 需要完整的 Git 历史来检测变更，因此使用了 `fetch-depth: 0`

5. **脚本权限**: `.github/scripts/detect_changed_tests.sh` 需要在 workflow 中设置执行权限（已在 ci.yml 中处理）

## 故障排查

### 问题: CI 没有找到对应的测试文件
- **检查**: 确认测试文件命名是否符合上述规则
- **解决**: 重命名测试文件或手动运行所有测试

### 问题: 想强制测试所有文件
- **使用 ci.yml**: 直接运行 `ci.yml` workflow，它会测试所有文件
- **Push**: 默认使用 `ci.yml`，测试所有文件
- **PR**: 可以手动触发 `ci.yml` 来测试所有文件
- **禁用检测**: 在 `ci-changed-files.yml` 中设置 `enable_changed_files_detection = false`

### 问题: 变更检测不准确
- **检查**: 查看 CI 日志中的 "Get changed test files" 步骤输出
- **解决**: 确认 Git 历史完整，或手动指定测试文件

### 问题: 脚本执行失败
- **检查**: 确认 `.github/scripts/detect_changed_tests.sh` 文件存在且有执行权限
- **解决**: workflow 中已自动设置权限，如仍有问题，检查文件路径是否正确
