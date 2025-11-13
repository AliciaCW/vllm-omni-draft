# vllm-omni CI 生效范围分析

本文档详细说明 vllm-omni 仓库中各个 CI 工作流的触发条件和检查范围。

---

## 工作流概览

| 工作流               | 文件路径                                 | 主要功能           |
| -------------------- | ---------------------------------------- | ------------------ |
| **CI**               | `.github/workflows/ci.yml`               | 运行完整测试套件   |
| **pre-commit**       | `.github/workflows/pre-commit.yml`       | 代码风格和静态检查 |
| **CI Changed Files** | `.github/workflows/ci-changed-files.yml` | 只测试变更的文件   |
| **Auto Merge**       | `.github/workflows/automerge.yml`        | 自动合并 PR        |

---

## 1. CI (ci.yml)

### 触发条件
- ✅ **所有分支的 push** (`branches: ["**"]`)
- ✅ **PR 到 main 分支** (`pull_request: branches: [main]`)
- ✅ **手动触发** (`workflow_dispatch`)

### 检查范围
- **测试文件**: `tests/` 目录下的所有测试
- **测试命令**: `pytest tests/ -v -m "not slow"`
- **排除**: 标记为 `slow` 的测试
- **Python 版本**: 3.12, 3.13（矩阵测试）

### 检查的文件类型
- 所有 Python 测试文件（`test_*.py`, `*_test.py`）
- 不检查源代码文件本身，只运行测试

### 何时运行
- 每次推送到任何分支
- 创建或更新 PR（目标分支为 main）
- 手动在 GitHub Actions 页面触发

---

## 2. pre-commit (pre-commit.yml)

### 触发条件
- ✅ **所有 PR** (`pull_request`)
- ✅ **所有分支的 push** (`push: branches: ["**"]`)

### 检查范围
- **使用**: `pre-commit/action@v3.0.1`
- **参数**: `--all-files --hook-stage manual`
- **检查**: **整个代码库的所有文件**（不只是变更的文件）

### 执行的检查（来自 .pre-commit-config.yaml）

#### 1. **pre-commit-hooks** (v4.6.0)
- `check-yaml`: 检查 YAML 文件语法
  - **文件**: 所有 `.yaml`, `.yml` 文件
- `debug-statements`: 检测调试语句
  - **文件**: 所有 Python 文件
- `end-of-file-fixer`: 确保文件末尾有换行
  - **文件**: 所有文件
- `mixed-line-ending`: 统一行尾符为 LF
  - **文件**: 所有文件
- `trailing-whitespace`: 移除行尾空白
  - **文件**: 所有文件（Markdown 文件特殊处理）

#### 2. **isort** (5.13.2)
- **文件**: 所有 Python 文件 (`.py`)
- **功能**: 整理导入语句顺序

#### 3. **black** (24.10.0)
- **文件**: 所有 Python 文件 (`.py`)
- **功能**: 代码格式化

#### 4. **ruff** (v0.11.7)
- **文件**: 所有 Python 文件 (`.py`)
- **功能**: 
  - Linting（代码检查）
  - 自动修复部分问题
  - 输出 GitHub Actions 格式

#### 5. **typos** (v1.35.5)
- **文件**: 所有文本文件
- **功能**: 拼写检查

#### 6. **actionlint** (v1.7.7)
- **文件**: `.github/workflows/*.yml`, `.github/workflows/*.yaml`
- **功能**: 检查 GitHub Actions 工作流文件语法

### 何时运行
- 每次推送到任何分支
- 创建或更新任何 PR
- **注意**: 检查整个代码库，不是只检查变更的文件

---

## 3. CI Changed Files (ci-changed-files.yml)

### 触发条件
- ✅ **手动触发** (`workflow_dispatch`)
- ✅ **被其他工作流调用** (`workflow_call`)

### 检查范围
- **智能检测**: 根据变更的文件，只运行相关的测试
- **回退机制**: 如果检测失败，运行完整测试套件
- **Python 版本**: 3.12, 3.13（矩阵测试）

### 文件检测逻辑
- 检测变更的源代码文件（`.py`）
- 查找对应的测试文件（`test_*.py`, `*_test.py`）
- 只运行相关的测试

### 何时运行
- 手动在 GitHub Actions 页面触发
- 被其他工作流调用（目前没有被调用）

---

## 4. Auto Merge (automerge.yml)

### 触发条件
- PR 被添加标签 (`labeled`)
- PR 有新提交 (`synchronize`)
- PR 被打开 (`opened`)
- PR 准备审查 (`ready_for_review`)
- PR 被重新打开 (`reopened`)
- PR 有新的 review (`pull_request_review: submitted`)
- CI 检查完成 (`check_suite: completed`)

### 功能
- 自动合并符合条件的 PR
- **要求**: PR 必须有 `automerge` 标签
- **要求**: 所有状态检查必须通过
- **合并方式**: SQUASH
- **更新方式**: rebase

### 何时运行
- PR 相关事件发生时自动检查
- 不检查代码，只负责合并

---

## 关键问题回答

### Q1: `.pre-commit-config.yaml` 会在 GitHub Actions 上运行吗？

**答案：是的！**

- ✅ `pre-commit.yml` 工作流会读取 `.pre-commit-config.yaml`
- ✅ 使用 `pre-commit/action@v3.0.1` 执行配置中的所有 hooks
- ✅ 参数 `--all-files` 表示检查整个代码库
- ✅ 参数 `--hook-stage manual` 表示运行 manual 阶段的 hooks

### Q2: 哪些文件会被检查？

#### pre-commit 工作流检查的文件：
- ✅ **所有 Python 文件** (`.py`) - 通过 black, isort, ruff, debug-statements
- ✅ **所有 YAML 文件** (`.yaml`, `.yml`) - 通过 check-yaml
- ✅ **所有文本文件** - 通过 typos, trailing-whitespace, end-of-file-fixer
- ✅ **GitHub Actions 工作流文件** - 通过 actionlint
- ✅ **所有文件** - 通过 mixed-line-ending, end-of-file-fixer

#### CI 工作流检查的文件：
- ✅ **测试文件** (`tests/` 目录) - 运行 pytest

### Q3: 什么时候会触发这些检查？

| 事件            | ci.yml | pre-commit.yml | ci-changed-files.yml | automerge.yml |
| --------------- | ------ | -------------- | -------------------- | ------------- |
| Push 到任何分支 | ✅      | ✅              | ❌                    | ❌             |
| Push 到 main    | ✅      | ✅              | ❌                    | ❌             |
| 创建 PR         | ✅      | ✅              | ❌                    | ✅             |
| 更新 PR         | ✅      | ✅              | ❌                    | ✅             |
| 手动触发        | ✅      | ❌              | ✅                    | ❌             |

---

## 总结

### pre-commit 工作流
- **会在 GitHub Actions 上运行**
- **检查整个代码库的所有文件**（不只是变更的文件）
- **执行 `.pre-commit-config.yaml` 中配置的所有 hooks**
- **触发条件**: 所有 PR 和所有分支的 push

### CI 工作流
- **运行测试套件**
- **检查**: `tests/` 目录下的所有测试
- **触发条件**: 所有分支的 push、PR 到 main、手动触发

### 建议
1. **本地开发**: 使用 `pre-commit install` 安装 Git hooks，在提交前自动检查
2. **CI 验证**: GitHub Actions 会自动运行 pre-commit 检查，确保代码质量
3. **测试验证**: CI 工作流会自动运行测试，确保功能正常

---

## 验证方法

### 检查 pre-commit 是否在 GitHub Actions 上运行：
1. 访问 GitHub 仓库的 Actions 页面
2. 查看左侧工作流列表，找到 "pre-commit"
3. 查看是否有运行记录
4. 或者在 PR 中查看检查状态

### 本地测试 pre-commit：
```bash
# 检查所有文件
pre-commit run --all-files

# 检查特定 hook
pre-commit run ruff --all-files
```

