# vLLM-omni CI 工作流设计文档

## 项目对比分析

### vllm-main/.github 结构分析

#### Workflows（工作流）
| 文件 | 作用 | 是否参考 |
|------|------|----------|
| `pre-commit.yml` | **代码格式检查**：运行 pre-commit hooks（black、isort、flake8 等） | **[必须]** |
| `bc-lint.yml` | 向后兼容性检查（BC Linter），需要 PyTorch test-infra | **[否]** 暂不需要 |
| `issue_autolabel.yml` | Issue 自动标签（基于关键词） | **[可选]** 后期添加 |
| `stale.yml` | 自动标记和关闭过期 Issues/PRs | **[可选]** 后期添加 |
| `cleanup_pr_body.yml` | PR 描述自动清理 | **[否]** 非必需 |
| `add_label_automerge.yml` | 自动合并标签管理 | **[否]** 非必需 |
| `reminder_comment.yml` | PR 提醒评论 | **[否]** 非必需 |

#### 配置文件
| 文件 | 作用 | 是否参考 |
|------|------|----------|
| `mergify.yml` | PR 自动合并规则和标签管理 | **[可选]** 后期添加 |
| `PULL_REQUEST_TEMPLATE.md` | PR 提交模板 | **[已有]** |
| `CODEOWNERS` | 代码所有者配置 | **[可选]** |
| `dependabot.yml` | 依赖自动更新 | **[可选]** 后期添加 |
| `scale-config.yml` | Scale CI 配置 | **[否]** 暂不需要 |

---

### mindone-master/.github 结构分析

#### Workflows（工作流）
| 文件 | 作用 | 是否参考 |
|------|------|----------|
| `ci.yml` | **CI 测试**：多 Python 版本测试 + lint 检查 | **[必须]** |
| `docs.yml` | 文档构建和部署到 GitHub Pages | **[可选]** 如有文档系统 |
| `publish.yml` | PyPI 包发布（release 时触发） | **[可选]** 后期添加 |

#### 配置文件
| 文件 | 作用 | 是否参考 |
|------|------|----------|
| `PULL_REQUEST_TEMPLATE.md` | PR 提交模板 | **[已有]** |
| `CODEOWNERS` | 代码所有者配置 | **[可选]** 可选 |

---

## vLLM-omni 现状

### 已有文件
- **[已]** `.github/ISSUE_TEMPLATE/` - Issue 模板（已完善）
- **[已]** `.github/PULL_REQUEST_TEMPLATE.md` - PR 模板
- **[已]** `.pre-commit-config.yaml` - Pre-commit 配置

### 缺失文件
- **[否]** `.github/workflows/` - CI 工作流（**需要创建**）

---

## vLLM-omni CI 工作流设计方案

### 设计原则

1. **精简优先**：项目初期只包含最基础的 CI 功能
2. **渐进式扩展**：后续可根据需要添加更多工作流
3. **参考成熟做法**：结合 vllm-main 和 mindone 的优点
4. **成本控制**：避免需要 GPU 的测试，使用 mock 和轻量级测试

---

## Phase 1：基础 CI 工作流（必须实现）

### 1. `pre-commit.yml` - 代码质量检查

**触发时机**：
- PR 提交时
- 推送到 main 分支时

**功能**：
- 运行 pre-commit hooks（black、isort、flake8、yaml 检查等）
- 确保代码格式符合规范

**参考**：vllm-main 的 pre-commit.yml（使用 pre-commit/action）

**优先级**：**[最高]**

---

### 2. `ci.yml` - 基础 CI 测试

**触发时机**：
- PR 提交时
- 推送到 main 分支时

**功能**：
- **Python 版本测试**：Python 3.9, 3.10, 3.11（根据 pyproject.toml 要求）
- **依赖安装**：安装项目依赖和开发依赖
- **单元测试**：运行 pytest 测试套件
- **代码覆盖率**：生成覆盖率报告（可选）

**测试策略**：
- 使用 CPU 运行（避免 GPU 依赖）
- 使用 mock 替代真实模型加载
- 只运行快速测试（跳过 `@pytest.mark.slow` 标记的测试）

**参考**：mindone 的 ci.yml（结构简单，适合项目初期）

**优先级**：**[最高]**

---

## Phase 2：PR 自动化（已完成）

### 1. `automerge.yml` - PR 自动合并

**触发时机**：
- PR 状态变化（标签、同步、审阅等）
- CI 检查完成时

**功能**：
- 自动合并满足条件的 PR
- 合并条件：CI 通过 + reviewer 批准 + 无冲突

**参考**：`vllm-main/.github/workflows/add_label_automerge.yml`（简化版）

**关键配置**：
- 使用 `pascalgn/automerge-action` action
- 需要 `contents: write` 和 `pull-requests: write` 权限
- 监听 PR 和 check_suite 事件

---

### 2. `CODEOWNERS` - 自动分配 Reviewer

**功能**：
- 根据文件路径自动分配 PR reviewer
- GitHub 原生功能，无需 workflow

**参考**：
- `vllm-main/.github/CODEOWNERS` - 复杂规则示例
- `mindone-master/.github/CODEOWNERS` - 简化规则示例

**配置说明**：
- `* @owner1` - 默认 reviewer（所有文件）
- `/.github/ @owner1` - .github 目录的 reviewer
- `/tests/ @owner1` - tests 目录的 reviewer

**使用前需要修改**：
- 将 `@owner1` 替换为实际的 GitHub 用户名或团队名
- 根据项目结构添加更多路径规则

**注意事项**：
- CODEOWNERS 需要仓库管理员权限才能生效
- 文件路径使用 glob 模式匹配
- 最后匹配的规则优先级最高

---

## Phase 3：扩展工作流（可选，后期添加）

### 3. `test.yml` - 完整测试套件（可选）

**触发时机**：
- PR 提交时（手动触发或特定标签）
- 推送到 main 分支时

**功能**：
- 运行所有测试（包括集成测试）
- 多 Python 版本矩阵测试
- 测试结果汇总

**优先级**：**[中等]**

---

### 4. `docs.yml` - 文档构建（可选）

**触发时机**：
- PR 提交到 docs 相关文件时
- 推送到 main 分支时

**功能**：
- 构建文档
- 部署到 GitHub Pages（仅 main 分支）

**前提条件**：项目需要有文档系统（如 mkdocs、sphinx）

**优先级**：**[中等]**

---

### 5. `publish.yml` - PyPI 发布（可选）

**触发时机**：
- 创建 GitHub Release 时

**功能**：
- 构建 Python 包
- 发布到 PyPI

**前提条件**：需要配置 PyPI API token

**优先级**：**[低]**

---

## 目前最小 CI 配置

### 文件结构

```
vllm-omni-main/
├── .github/
│   ├── workflows/
│   │   ├── pre-commit.yml      # Phase 1: 代码质量检查
│   │   ├── ci.yml               # Phase 1: 基础 CI 测试
│   │   └── automerge.yml       # Phase 2: PR 自动合并
│   ├── CODEOWNERS               # Phase 2: 自动分配 reviewer
│   ├── ISSUE_TEMPLATE/          # [旧] 原始已有
│   └── PULL_REQUEST_TEMPLATE.md # [旧] 原始已有
├── .pre-commit-config.yaml      # pre-commit 配置
├── pyproject.toml               # 项目配置
└── tests/
    ├── __init__.py
    └── test_example.py          # 示例测试
```

---

## 详细设计：pre-commit.yml

```yaml
name: pre-commit

on:
  pull_request:
  push:
    branches: [main]

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: ${{ github.event_name == 'pull_request' }}

permissions:
  contents: read

jobs:
  pre-commit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - uses: pre-commit/action@v3.0.1
        with:
          extra_args: --all-files --hook-stage manual
```

**关键配置**：
- 使用固定版本的 actions（避免破坏性更新）
- 支持并发取消（PR 更新时取消旧运行）
- 运行所有文件的检查

---

## 详细设计：ci.yml

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

permissions:
  contents: read

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11"]
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
      
      - name: Run tests
        run: |
          pytest tests/ -v --cov=vllm_omni --cov-report=xml -m "not slow"
      
      - name: Upload coverage
        uses: codecov/codecov-action@v4
        if: matrix.python-version == '3.11'
        with:
          file: ./coverage.xml
          fail_ci_if_error: false
```

**关键配置**：
- **多 Python 版本**：测试兼容性
- **跳过慢测试**：`-m "not slow"` 只运行快速测试
- **覆盖率报告**：仅在一个 Python 版本上传（避免重复）；失败不影响 CI（`fail_ci_if_error: false`）

---

## 详细设计：automerge.yml

```yaml
name: Auto Merge PR

on:
  pull_request:
    types:
      - labeled
      - synchronize
      - opened
      - ready_for_review
      - reopened
  pull_request_review:
    types:
      - submitted
  check_suite:
    types:
      - completed

permissions:
  contents: write
  pull-requests: write

jobs:
  automerge:
    runs-on: ubuntu-latest
    steps:
      - name: Auto Merge
        uses: pascalgn/automerge-action@v0.16.4
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

**关键配置**：
- 监听 PR 状态变化和 CI 检查完成
- 需要写入权限才能合并 PR
- 自动合并条件：CI 通过 + reviewer 批准 + 无冲突

---

## 详细设计：CODEOWNERS

```
# CODEOWNERS 文件用于自动分配 PR reviewer
# 格式：路径 pattern @用户名或团队名

# 默认所有者（所有文件的默认 reviewer）
* @owner1

# 按目录分配 reviewer
/.github/ @owner1
/tests/ @owner1
/docs/ @owner1
```

**关键配置**：
- 使用 glob 模式匹配文件路径
- 最后匹配的规则优先级最高
- 支持团队名和用户名

---

## 实施步骤

### Step 1：创建 workflows 目录
```bash
mkdir -p vllm-omni-main/.github/workflows
```

### Step 2：创建 pre-commit.yml
- 复制参考配置
- 根据项目需要调整 Python 版本

### Step 3：创建 ci.yml
- 根据 pyproject.toml 中的 Python 版本要求配置
- 根据测试目录结构调整 pytest 命令
- 配置覆盖率报告（可选）

### Step 4：创建支持文件
- `.pre-commit-config.yaml` - pre-commit 配置（pre-commit.yml 需要）
- `pyproject.toml` - 项目配置（ci.yml 安装依赖需要）
- `tests/__init__.py` - pytest 测试目录标识
- `tests/test_example.py` - 示例测试文件

### Step 5：创建 PR 自动化文件
- `.github/CODEOWNERS` - 自动分配 reviewer（需要修改用户名）
- `.github/workflows/automerge.yml` - PR 自动合并

### Step 6：测试验证
- 创建测试 PR，验证工作流正常运行
- 检查测试结果
- 验证 reviewer 自动分配
- 验证自动合并功能

---

## 支持文件说明

### `.pre-commit-config.yaml`
pre-commit 配置文件，定义代码检查规则：
- `pre-commit-hooks`：基础检查（yaml、文件末尾、空白符等）
- `isort`：导入排序
- `black`：代码格式化
- `flake8`：代码风格检查

### `pyproject.toml`
项目配置文件，包含：
- 项目基本信息
- 开发依赖（pytest、black、isort、flake8 等）
- pytest 配置（测试路径、标记等）

### `tests/__init__.py`
空文件，标识 `tests/` 为 Python 包，pytest 才能识别。

### `tests/test_example.py`
示例测试文件：
- `test_example()`：基础测试，CI 会运行
- `test_slow_example()`：标记为慢测试，CI 会跳过

---

## 注意事项

### 1. Actions 版本固定
- 使用固定版本号（如 `@v4`），避免自动更新导致的破坏性变更
- 参考 vllm-main 的做法，使用 commit SHA（更安全）

### 2. 测试时间控制
- CI 测试应在 10 分钟内完成
- 使用 `-m "not slow"` 跳过需要 GPU 或长时间运行的测试
- 避免在 CI 中下载大模型

### 3. 成本控制
- 使用 GitHub 免费额度（2000 分钟/月）
- 避免在多个分支重复运行
- 使用 concurrency 配置避免重复运行

### 4. 安全考虑
- 不要将敏感信息（API keys、tokens）硬编码
- 使用 GitHub Secrets 存储敏感配置
- 限制工作流的权限（使用 `permissions` 字段）

### 5. PR 自动化注意事项
- CODEOWNERS 需要仓库管理员权限才能生效
- automerge-action 需要 `contents: write` 和 `pull-requests: write` 权限
- 自动合并前确保 CI 检查配置正确
- 建议在项目初期先手动合并，验证流程后再启用自动合并

---

## 后续扩展建议

### 短期（1-3 个月）
1. 添加 `test.yml`（完整测试套件，可选触发）
2. 添加 Issue 自动标签（简化维护）
3. [已完成] `CODEOWNERS` 和 `automerge.yml`（PR 自动分配和合并）

### 中期（3-6 个月）
1. 添加 `docs.yml`（文档自动构建和部署）
2. 添加 `publish.yml`（PyPI 自动发布）
3. 添加 `stale.yml`（自动管理过期 Issues/PRs）

### 长期（6 个月+）
1. 添加 GPU 测试（如果有自建 runner）
2. 添加性能基准测试
3. 添加 dependabot（依赖自动更新）

---

## 参考资源

- [GitHub Actions 文档](https://docs.github.com/zh/actions)
- [Pre-commit 文档](https://pre-commit.com/)
- [Pytest 文档](https://docs.pytest.org/)
- [vLLM CI 文档](https://docs.vllm.ai/en/latest/contributing)
- [CODEOWNERS 文档](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)

---

## 总结

### 基础 CI 配置

**工作流文件**：
1. **pre-commit.yml** - 代码质量检查
2. **ci.yml** - 基础测试（多 Python 版本）
3. **automerge.yml** - PR 自动合并

**配置文件**：
1. **CODEOWNERS** - 自动分配 reviewer
2. **.pre-commit-config.yaml** - pre-commit 配置
3. **pyproject.toml** - 项目配置和依赖

**测试文件**：
1. **tests/__init__.py** - 测试目录标识
2. **tests/test_example.py** - 示例测试

### 方案对比
- **vllm-main**：功能全面，适合大型项目
- **mindone**：精简实用，适合项目初期
- **vllm-omni 方案**：参考 mindone 的简洁性，结合 vllm-main 的 pre-commit 做法

### 实施优先级
- ** Phase 1 （已完成）**：pre-commit.yml + ci.yml
- ** Phase 2（已完成）**：automerge.yml + CODEOWNERS
- ** Phase 3 **：test.yml + Issue 标签
- ** Phase 4 **：docs.yml + publish.yml





# 附录： PR 自动分配 Reviewer 和自动合并配置说明

## 参考文件

### 1. CODEOWNERS
- **参考来源**：`vllm-main/.github/CODEOWNERS`、`mindone-master/.github/CODEOWNERS`
- **功能**：GitHub 原生功能，根据文件路径自动分配 PR reviewer
- **使用方式**：GitHub 会自动读取 `.github/CODEOWNERS` 文件，PR 创建时自动请求对应 reviewer

### 2. automerge.yml
- **参考来源**：`vllm-main/.github/workflows/add_label_automerge.yml`（简化版）
- **功能**：使用 GitHub Actions 自动合并满足条件的 PR
- **合并条件**：
  - CI 检查通过
  - 至少一个 reviewer 批准
  - 没有冲突
  - 没有 blocked 标签

## 文件说明

### `.github/CODEOWNERS`
自动分配 reviewer 规则：
- `* @owner1` - 默认 reviewer（所有文件）
- `/.github/ @owner1` - .github 目录的 reviewer
- `/tests/ @owner1` - tests 目录的 reviewer

**使用前需要修改**：
- 将 `@owner1` 替换为实际的 GitHub 用户名或团队名
- 根据项目结构添加更多路径规则

### `.github/workflows/automerge.yml`
自动合并工作流：
- 监听 PR 状态变化（标签、同步、审阅等）
- 监听 CI 检查完成
- 当满足条件时自动合并 PR

**依赖**：
- 需要 GitHub Actions 权限
- 使用 `pascalgn/automerge-action` action

## 使用说明

1. **配置 CODEOWNERS**：
   - 修改 `.github/CODEOWNERS` 中的用户名
   - 添加项目特定的路径规则

2. **启用自动合并**：
   - 文件已创建，GitHub Actions 会自动运行
   - 可在仓库 Settings > Actions 中查看工作流状态

3. **可选配置**：
   - 如需更复杂的合并规则，可参考 `vllm-main/.github/mergify.yml` 配置 Mergify
   - Mergify 需要安装 Mergify App，功能更强大但需要额外配置

## 注意事项

- CODEOWNERS 需要仓库管理员权限才能生效
- automerge-action 需要 `contents: write` 和 `pull-requests: write` 权限
- 自动合并前确保 CI 检查配置正确
- 建议在项目初期先手动合并，验证流程后再启用自动合并
