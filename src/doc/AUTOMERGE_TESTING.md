# Auto Merge Workflow 验证指南

`automerge.yml` 需要真实的 GitHub 环境才能运行，无法完全在本地测试。本文档提供验证和测试方法。

## 为什么无法本地测试？

1. **需要 GitHub API**: automerge action 需要访问 GitHub API 来合并 PR
2. **需要 PR 环境**: 需要真实的 PR、标签、reviewer 等 GitHub 功能
3. **需要权限**: 需要 `contents: write` 和 `pull-requests: write` 权限

## 验证方法

### 方法 1: 语法和配置检查

#### 1.1 检查 YAML 语法

```bash
# 使用 yamllint 或在线工具检查语法
yamllint .github/workflows/automerge.yml

# 或使用 GitHub Actions 的语法检查
# 推送到 GitHub 后，Actions 会自动验证语法
```

#### 1.2 验证配置项

检查以下配置是否正确：

- ✅ **Action 版本**: `pascalgn/automerge-action@v0.16.4` (使用固定版本，避免破坏性更新)
- ✅ **权限**: `contents: write` 和 `pull-requests: write` (必需)
- ✅ **环境变量**:
  - `GITHUB_TOKEN`: 自动提供，无需配置
  - `MERGE_METHOD`: SQUASH (合并方式)
  - `UPDATE_METHOD`: rebase (更新方式)
  - `REQUIRED_LABELS`: automerge (必需标签)
  - `REQUIRED_STATUS_CHECKS`: true (需要所有检查通过)

### 方法 2: 在测试仓库中验证

#### 2.1 创建测试仓库

1. 创建一个测试仓库（或使用现有的测试分支）
2. 将 `automerge.yml` 推送到测试仓库
3. 创建测试 PR

#### 2.2 测试步骤

```bash
# 1. 创建测试分支
git checkout -b test-automerge
git add .github/workflows/automerge.yml
git commit -m "Add automerge workflow"
git push origin test-automerge

# 2. 在 GitHub 上创建 PR
# 3. 添加 'automerge' 标签
# 4. 确保所有 CI 检查通过
# 5. 等待 automerge action 运行
```

#### 2.3 验证清单

- [ ] Workflow 文件语法正确（GitHub Actions 页面无错误）
- [ ] Workflow 在 PR 事件时触发
- [ ] 添加 `automerge` 标签后，workflow 运行
- [ ] 所有 CI 检查通过后，PR 被合并
- [ ] 合并方式为 SQUASH
- [ ] 合并前 PR 分支被 rebase 更新

### 方法 3: 查看 GitHub Actions 日志

1. 访问 GitHub 仓库的 Actions 页面
2. 查看 "Auto Merge PR" workflow 的运行记录
3. 检查日志输出：
   - 是否有错误信息
   - 是否成功检测到 PR
   - 是否满足合并条件
   - 是否成功合并

### 方法 4: 使用 GitHub CLI 模拟（部分验证）

可以使用 GitHub CLI 来验证配置，但无法完全模拟合并过程：

```bash
# 安装 GitHub CLI
brew install gh  # macOS
# 或访问 https://cli.github.com/

# 登录
gh auth login

# 查看 workflow 文件
gh workflow view automerge.yml

# 查看 workflow 运行历史
gh run list --workflow=automerge.yml
```

## 配置验证清单

### ✅ 基本配置

- [ ] Workflow 名称正确
- [ ] 触发事件配置正确（pull_request, pull_request_review, check_suite）
- [ ] 权限配置正确（contents: write, pull-requests: write）

### ✅ Action 配置

- [ ] 使用固定版本的 action (`@v0.16.4`)
- [ ] GITHUB_TOKEN 正确配置（自动提供）
- [ ] MERGE_METHOD 设置为 SQUASH
- [ ] UPDATE_METHOD 设置为 rebase
- [ ] REQUIRED_LABELS 设置为 automerge
- [ ] REQUIRED_STATUS_CHECKS 设置为 true

### ✅ 安全配置

- [ ] 只合并带有 `automerge` 标签的 PR
- [ ] 要求所有状态检查通过
- [ ] 使用 SQUASH 合并（保持历史整洁）

## 常见问题排查

### 问题 1: Workflow 未触发

**检查**:
- PR 是否创建/更新
- 是否添加了 `automerge` 标签
- Workflow 文件是否在正确的路径（`.github/workflows/`）

**解决**:
- 检查 GitHub Actions 页面是否有错误
- 确认 workflow 文件已提交到仓库

### 问题 2: PR 未自动合并

**检查**:
- 是否所有 CI 检查通过
- 是否有 reviewer 批准（如果仓库要求）
- PR 是否有冲突

**解决**:
- 查看 workflow 日志了解原因
- 检查 `REQUIRED_STATUS_CHECKS` 设置
- 确认所有必需的检查都已通过

### 问题 3: 合并方式不正确

**检查**:
- `MERGE_METHOD` 设置是否正确
- GitHub 仓库设置是否允许该合并方式

**解决**:
- 确认 `MERGE_METHOD: SQUASH` 设置正确
- 检查仓库的合并设置

## 测试建议

### 首次使用前

1. **在测试分支验证**:
   - 创建测试 PR
   - 添加 `automerge` 标签
   - 观察 workflow 行为

2. **小范围测试**:
   - 先在一个不重要的 PR 上测试
   - 确认行为符合预期后再广泛使用

3. **监控日志**:
   - 首次使用时密切关注日志
   - 确认合并行为正确

### 生产环境使用

1. **设置保护规则**:
   - 在 GitHub 仓库设置中配置分支保护
   - 确保重要分支不会被意外合并

2. **团队沟通**:
   - 告知团队成员 automerge 功能
   - 说明如何使用 `automerge` 标签

3. **文档化**:
   - 在项目文档中说明 automerge 的使用方法
   - 记录合并策略和规则

## 参考资源

- [pascalgn/automerge-action 文档](https://github.com/pascalgn/automerge-action)
- [GitHub Actions 文档](https://docs.github.com/en/actions)
- [GitHub CLI 文档](https://cli.github.com/manual/)

## 快速验证命令

```bash
# 检查 YAML 语法（需要安装 yamllint）
yamllint .github/workflows/automerge.yml

# 检查文件是否存在
test -f .github/workflows/automerge.yml && echo "File exists" || echo "File not found"

# 查看 workflow 内容
cat .github/workflows/automerge.yml

# 使用 GitHub CLI 查看（如果已安装）
gh workflow view automerge.yml
```

