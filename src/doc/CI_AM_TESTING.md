# GitHub 私有库测试方案

本文档提供在 GitHub 私有库中测试 `automerge.yml` 和 `ci-all-files.yml` 工作流的详细方案。

## 测试目标

1. **automerge.yml**: 验证至少一个 reviewer 批准后自动合入功能
2. **ci-all-files.yml**: 验证手动触发的完整测试套件功能

## 前置准备

### 1. 创建测试仓库

1. 在 GitHub 上创建一个新的私有仓库（例如：`vllm-omni-test`）
2. 确保有至少两个 GitHub 账号（一个作为 PR 创建者，一个作为 reviewer）

### 2. 仓库设置

#### 2.1 启用 GitHub Actions

- 进入仓库 Settings → Actions → General
- 确保 "Allow all actions and reusable workflows" 已启用
- 确保 "Workflow permissions" 设置为 "Read and write permissions"

#### 2.2 分支保护规则（可选但推荐）

- 进入 Settings → Branches
- 为 `main` 分支添加保护规则：
  - 要求 PR 审查（Require pull request reviews before merging）
  - 至少需要 1 个批准（Required number of approvals: 1）
  - 要求状态检查通过（Require status checks to pass before merging）

### 3. 准备测试代码

创建最小化的项目结构用于测试：

```bash
# 克隆测试仓库
git clone https://github.com/YOUR_USERNAME/vllm-omni-test.git
cd vllm-omni-test

# 创建基本项目结构
mkdir -p .github/workflows
mkdir -p .github/scripts
mkdir -p tests
mkdir -p vllm_omni
```

## 测试方案 1: automerge.yml

### 步骤 1: 准备测试文件

#### 1.1 复制工作流文件

```bash
# 从主项目复制 automerge.yml
cp /path/to/vllm-omni/.github/workflows/automerge.yml \
   .github/workflows/automerge.yml
```

#### 1.2 创建最小化的项目文件

创建 `pyproject.toml`:

```toml
[project]
name = "vllm-omni-test"
version = "0.1.0"
dependencies = []

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
]
```

创建 `tests/test_example.py`:

```python
"""Simple test file for CI validation."""

def test_example():
    """Basic test that always passes."""
    assert True

def test_slow_example():
    """Slow test that should be skipped."""
    import pytest
    pytest.mark.slow
    assert True
```

创建 `tests/__init__.py` (空文件)

#### 1.3 创建 CI 工作流（用于触发 automerge）

创建 `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

permissions:
  contents: read

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
      - name: Run tests
        run: |
          pytest tests/ -v -m "not slow"
```

### 步骤 2: 初始提交

```bash
git add .
git commit -m "Initial commit: Add test files and workflows"
git push origin main
```

### 步骤 3: 创建测试 PR

#### 3.1 创建测试分支和修改

```bash
# 创建测试分支
git checkout -b test-automerge-feature

# 创建一个简单的修改
echo "# Test Feature" >> README.md
git add README.md
git commit -m "Add test feature"
git push origin test-automerge-feature
```

#### 3.2 在 GitHub 上创建 PR

1. 访问测试仓库的 GitHub 页面
2. 点击 "New Pull Request"
3. 选择 `test-automerge-feature` → `main`
4. 填写 PR 标题和描述（例如："Test automerge with reviewer approval"）

### 步骤 4: 测试场景

#### 场景 4.1: 无标签，无批准（不应合并）

1. **操作**: 创建 PR，不添加 `automerge` 标签，不批准
2. **预期结果**: 
   - automerge workflow 可能触发，但不会合并
   - PR 保持打开状态
3. **验证**: 检查 GitHub Actions 日志，确认未合并

#### 场景 4.2: 有标签，无批准（不应合并）

1. **操作**: 添加 `automerge` 标签，等待 CI 通过，但不批准
2. **预期结果**: 
   - automerge workflow 触发
   - 检测到缺少批准，不合并
   - PR 保持打开状态
3. **验证**: 
   - 查看 automerge workflow 日志
   - 确认日志显示 "Waiting for required approvals"

#### 场景 4.3: 有标签，有批准，CI 未通过（不应合并）

1. **操作**: 
   - 添加 `automerge` 标签
   - 让 reviewer 批准 PR
   - 故意让 CI 失败（例如：在测试中添加 `assert False`）
2. **预期结果**: 
   - automerge workflow 触发
   - 检测到 CI 未通过，不合并
   - PR 保持打开状态
3. **验证**: 
   - 查看 automerge workflow 日志
   - 确认日志显示 "Waiting for required status checks"

#### 场景 4.4: 有标签，有批准，CI 通过（应合并）✅

1. **操作**: 
   - 添加 `automerge` 标签
   - 等待 CI 通过
   - 让 reviewer 批准 PR
2. **预期结果**: 
   - automerge workflow 触发
   - 检测到所有条件满足（标签、批准、CI 通过）
   - PR 自动合并（使用 SQUASH 方式）
3. **验证**: 
   - 查看 automerge workflow 日志，确认合并成功
   - 检查 PR 状态，确认已合并
   - 检查合并提交，确认使用 SQUASH 方式

#### 场景 4.5: 多个 reviewer 批准（应合并）

1. **操作**: 
   - 添加 `automerge` 标签
   - 等待 CI 通过
   - 让多个 reviewer 批准 PR（至少 1 个即可）
2. **预期结果**: 
   - automerge workflow 触发
   - 检测到至少 1 个批准，满足条件
   - PR 自动合并
3. **验证**: 
   - 确认只需要 1 个批准即可合并
   - 多个批准不影响合并逻辑

### 步骤 5: 验证合并结果

合并后检查：

1. **合并方式**: 确认使用 SQUASH 合并（所有提交合并为一个）
2. **提交信息**: 检查合并提交的格式
3. **分支状态**: 确认源分支可以安全删除
4. **工作流日志**: 查看完整的 automerge workflow 执行日志

## 测试方案 2: ci-all-files.yml

### 步骤 1: 准备测试文件

#### 1.1 复制工作流文件

```bash
# 从主项目复制 ci-all-files.yml
cp /path/to/vllm-omni/.github/workflows/ci-all-files.yml \
   .github/workflows/ci-all-files.yml
```

#### 1.2 确保项目结构完整

确保有以下文件：
- `pyproject.toml` (包含依赖配置)
- `tests/test_example.py` (测试文件)
- `tests/__init__.py`

### 步骤 2: 测试场景

#### 场景 2.1: 手动触发工作流

1. **操作**: 
   - 访问 GitHub 仓库的 Actions 页面
   - 选择 "CI Changed Files" workflow
   - 点击 "Run workflow" 按钮
   - 选择分支（例如：`main`）
   - 点击 "Run workflow"
2. **预期结果**: 
   - 工作流开始运行
   - 显示运行状态和日志
3. **验证**: 
   - 检查工作流是否成功启动
   - 查看运行日志

#### 场景 2.2: 验证 Python 版本矩阵

1. **操作**: 手动触发工作流
2. **预期结果**: 
   - 工作流为每个 Python 版本（3.12, 3.13）创建独立的 job
   - 两个 job 并行运行
3. **验证**: 
   - 在 Actions 页面查看 job 列表
   - 确认有两个 job（Python 3.12 和 3.13）
   - 检查每个 job 的日志

#### 场景 2.3: 验证 uv 环境配置

1. **操作**: 手动触发工作流
2. **预期结果**: 
   - "Prepare environment with uv" 步骤成功
   - 虚拟环境正确创建
   - 依赖正确安装
3. **验证**: 
   - 查看 "Prepare environment with uv" 步骤的日志
   - 确认 `uv venv` 命令成功执行
   - 确认 `uv pip install` 成功安装依赖
   - 检查虚拟环境路径是否正确设置

#### 场景 2.4: 验证测试执行

1. **操作**: 手动触发工作流
2. **预期结果**: 
   - "Run tests" 步骤成功执行
   - 测试文件被正确发现和执行
   - 测试通过
3. **验证**: 
   - 查看 "Run tests" 步骤的日志
   - 确认 pytest 正确发现测试文件
   - 确认测试执行结果
   - 确认慢测试被正确跳过（`-m "not slow"`）

#### 场景 2.5: 无测试文件时的行为

1. **操作**: 
   - 临时删除或重命名 `tests/` 目录
   - 手动触发工作流
2. **预期结果**: 
   - "Run tests" 步骤检测到无测试文件
   - 输出提示信息
   - 步骤成功退出（不失败）
3. **验证**: 
   - 查看 "Run tests" 步骤的日志
   - 确认输出 "No pytest-compatible test files detected"
   - 确认工作流成功完成（绿色）

#### 场景 2.6: 测试失败时的行为

1. **操作**: 
   - 在测试文件中添加会失败的测试（例如：`assert False`）
   - 手动触发工作流
2. **预期结果**: 
   - 测试执行
   - 测试失败
   - 工作流标记为失败
3. **验证**: 
   - 查看测试输出，确认失败原因
   - 确认工作流状态为失败（红色）

### 步骤 3: 验证工作流配置

检查以下配置项：

- ✅ **触发方式**: 仅 `workflow_dispatch`（手动触发）
- ✅ **Python 版本矩阵**: 3.12, 3.13
- ✅ **超时设置**: 30 分钟
- ✅ **磁盘空间清理**: 执行清理步骤
- ✅ **uv 环境**: 正确配置和使用
- ✅ **测试执行**: 正确发现和执行测试

## 测试检查清单

### automerge.yml 测试清单

- [ ] 工作流文件语法正确
- [ ] 工作流在 PR 事件时触发
- [ ] 无 `automerge` 标签时，PR 不合并
- [ ] 有标签但无批准时，PR 不合并
- [ ] 有标签和批准但 CI 未通过时，PR 不合并
- [ ] 有标签、批准和 CI 通过时，PR 自动合并
- [ ] 合并方式为 SQUASH
- [ ] 只需要 1 个 reviewer 批准即可合并
- [ ] 工作流日志清晰，便于排查问题

### ci-all-files.yml 测试清单

- [ ] 工作流文件语法正确
- [ ] 可以手动触发工作流
- [ ] Python 版本矩阵正确（3.12, 3.13）
- [ ] uv 环境正确配置
- [ ] 依赖正确安装
- [ ] 测试文件正确发现
- [ ] 测试正确执行
- [ ] 慢测试被正确跳过
- [ ] 无测试文件时优雅处理（不失败）
- [ ] 测试失败时工作流正确标记为失败
- [ ] 工作流日志清晰，便于排查问题

## 常见问题排查

### automerge.yml 问题

#### 问题 1: PR 未自动合并

**检查项**:
- PR 是否有 `automerge` 标签
- 是否有至少 1 个 reviewer 批准
- 所有 CI 检查是否通过
- 工作流是否触发并运行

**排查步骤**:
1. 查看 PR 页面，确认标签和批准状态
2. 查看 Actions 页面，确认工作流运行状态
3. 查看工作流日志，查找错误信息
4. 检查分支保护规则是否阻止合并

#### 问题 2: 工作流未触发

**检查项**:
- 工作流文件是否在 `.github/workflows/` 目录
- 工作流文件是否已提交到仓库
- PR 事件是否匹配触发条件

**排查步骤**:
1. 确认文件路径正确
2. 检查 GitHub Actions 是否启用
3. 查看 Actions 页面是否有错误提示

### ci-all-files.yml 问题

#### 问题 1: 工作流无法手动触发

**检查项**:
- 工作流文件语法是否正确
- GitHub Actions 是否启用
- 是否有权限触发工作流

**排查步骤**:
1. 检查 YAML 语法
2. 确认仓库设置中 Actions 已启用
3. 尝试在 Actions 页面手动触发

#### 问题 2: uv 环境配置失败

**检查项**:
- Python 版本是否正确
- uv 是否正确安装
- 虚拟环境路径是否正确

**排查步骤**:
1. 查看 "Prepare environment with uv" 步骤日志
2. 确认 Python 版本匹配
3. 检查虚拟环境创建命令

#### 问题 3: 测试未执行

**检查项**:
- 测试文件是否存在
- 测试文件命名是否正确
- pytest 配置是否正确

**排查步骤**:
1. 查看 "Run tests" 步骤日志
2. 确认测试文件路径和命名
3. 检查 `pyproject.toml` 中的 pytest 配置

## 测试报告模板

测试完成后，记录以下信息：

### 测试环境

- **测试仓库**: [仓库 URL]
- **测试日期**: [日期]
- **测试人员**: [姓名]
- **GitHub 账号**: [账号名]

### automerge.yml 测试结果

| 测试场景                  | 预期结果 | 实际结果 | 状态 | 备注 |
| ------------------------- | -------- | -------- | ---- | ---- |
| 无标签，无批准            | 不合并   |          |      |      |
| 有标签，无批准            | 不合并   |          |      |      |
| 有标签，有批准，CI 未通过 | 不合并   |          |      |      |
| 有标签，有批准，CI 通过   | 自动合并 |          |      |      |
| 多个 reviewer 批准        | 自动合并 |          |      |      |

### ci-all-files.yml 测试结果

| 测试场景        | 预期结果      | 实际结果 | 状态 | 备注 |
| --------------- | ------------- | -------- | ---- | ---- |
| 手动触发        | 工作流运行    |          |      |      |
| Python 版本矩阵 | 两个 job 并行 |          |      |      |
| uv 环境配置     | 环境正确创建  |          |      |      |
| 测试执行        | 测试正确运行  |          |      |      |
| 无测试文件      | 优雅处理      |          |      |      |
| 测试失败        | 工作流失败    |          |      |      |

### 问题记录

记录测试过程中发现的问题：

1. **问题描述**: 
   - **复现步骤**: 
   - **预期行为**: 
   - **实际行为**: 
   - **解决方案**: 

### 结论

- [ ] automerge.yml 功能正常
- [ ] ci-all-files.yml 功能正常
- [ ] 所有测试场景通过
- [ ] 可以部署到生产环境

## 参考资源

- [pascalgn/automerge-action 文档](https://github.com/pascalgn/automerge-action)
- [GitHub Actions 文档](https://docs.github.com/en/actions)
- [uv 文档](https://github.com/astral-sh/uv)
- [pytest 文档](https://docs.pytest.org/)

