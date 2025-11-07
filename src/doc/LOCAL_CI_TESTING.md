# 本地 CI 测试指南

本文档介绍如何在本地测试 GitHub Actions 工作流，分为三个步骤：
1. 本地模拟 CI 命令（直接运行 pytest、pre-commit）**← 不需要 Docker**
2. 使用 act 工具本地运行 GitHub Actions **← 需要 Docker**
3. 推送到 GitHub 进行真实测试 **← 不需要 Docker**

> **没有 Docker？** 可以跳过步骤 2，直接使用步骤 1 和步骤 3 进行测试。步骤 1 已经能够验证大部分 CI 功能。

## 步骤 1: 本地模拟 CI 命令

### 快速开始

使用提供的测试脚本：

```bash
# 运行所有测试
chmod +x test_ci_local.sh
./test_ci_local.sh all

# 只运行 CI 测试
./test_ci_local.sh ci

# 只运行 pre-commit 检查
./test_ci_local.sh pre-commit
```

### 手动执行

#### 1.1 模拟 ci.yml

```bash
# 进入项目目录
cd vllm-omni

# 升级 pip
python3 -m pip install --upgrade pip

# 安装开发依赖
pip install -e ".[dev]"

# 运行测试（排除 slow 标记）
pytest tests/ -v -m "not slow"
```

#### 1.2 模拟 pre-commit.yml

```bash
# 安装 pre-commit（如果未安装）
pip install pre-commit

# 安装 hooks
pre-commit install

# 运行检查（模拟 GitHub Actions 的行为）
pre-commit run --all-files --hook-stage manual
```

## 步骤 2: 使用 act 本地运行 GitHub Actions（可选）

> **注意**: 此步骤需要 Docker。如果没有 Docker，可以跳过此步骤，直接进行步骤 3（推送到 GitHub 测试）。

[act](https://github.com/nektos/act) 是一个工具，可以在本地运行 GitHub Actions。它使用 Docker 容器来模拟 GitHub Actions 的运行环境。

### 2.1 安装 act

**macOS:**
```bash
brew install act
```

**Linux:**
```bash
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
```

**其他方式:** 参考 [act 官方文档](https://github.com/nektos/act#installation)

### 2.2 使用测试脚本

```bash
# 运行所有 workflows
chmod +x test_with_act.sh
./test_with_act.sh all

# 运行特定 workflow
./test_with_act.sh ci push
./test_with_act.sh pre-commit pull_request
```

### 2.3 手动使用 act

#### 运行 ci.yml

```bash
# 模拟 push 事件
act push -W .github/workflows/ci.yml --container-architecture linux/amd64

# 模拟 pull_request 事件
act pull_request -W .github/workflows/ci.yml --container-architecture linux/amd64

# 手动触发 (workflow_dispatch)
act workflow_dispatch -W .github/workflows/ci.yml --container-architecture linux/amd64
```

#### 运行 pre-commit.yml

```bash
act push -W .github/workflows/pre-commit.yml --container-architecture linux/amd64
```

#### 运行特定 job

```bash
# 只运行 test job
act push -W .github/workflows/ci.yml -j test --container-architecture linux/amd64
```

### 2.4 act 常用选项

- `-W, --workflows`: 指定 workflow 文件路径
- `-e, --eventpath`: 指定事件 JSON 文件
- `-j, --job`: 只运行指定的 job
- `--container-architecture`: 指定容器架构（M1/M2 Mac 建议用 `linux/amd64`）
- `-l, --list`: 列出所有可用的 jobs
- `--dryrun`: 干运行，不实际执行

### 2.5 注意事项

1. **需要 Docker**: act 依赖 Docker 来运行容器，必须安装并运行 Docker
2. **首次运行较慢**: act 需要下载 Docker 镜像，首次运行会较慢
3. **网络问题**: 某些 actions（如 `actions/checkout`）可能需要网络访问
4. **automerge.yml**: 需要真实的 GitHub token，本地测试可能无法完全模拟

### 2.6 没有 Docker 怎么办？

如果没有 Docker，可以：
- **跳过步骤 2**：步骤 1（本地模拟 CI 命令）已经能够验证大部分功能
- **直接进行步骤 3**：推送到 GitHub 进行真实测试，GitHub Actions 会在云端运行
- **安装 Docker**（可选）：
  - macOS: 下载 [Docker Desktop](https://www.docker.com/products/docker-desktop/)
  - Linux: `sudo apt-get install docker.io` 或使用包管理器安装

## 步骤 3: 推送到 GitHub 测试

### 3.1 创建测试分支

```bash
# 创建测试分支
git checkout -b test-ci-workflows

# 提交更改
git add .github/workflows/
git commit -m "Add CI workflows"

# 推送到远程
git push origin test-ci-workflows
```

### 3.2 触发 workflows

#### 触发 ci.yml

- **方式 1**: Push 到 `dev_CI` 分支
- **方式 2**: 创建 PR 到 `main` 分支
- **方式 3**: 在 GitHub 网页上手动触发（workflow_dispatch）

#### 触发 pre-commit.yml

- Push 到 `main` 分支
- 创建 PR

#### 触发 automerge.yml

- 创建 PR 并添加 `automerge` 标签
- 确保所有检查通过

### 3.3 查看运行结果

1. 访问 GitHub 仓库页面
2. 点击 "Actions" 标签
3. 查看 workflow 运行状态和日志

## 没有 Docker 的测试流程

如果没有 Docker，推荐以下测试流程：

1. **使用步骤 1** 验证代码和配置：
   ```bash
   ./test_ci_local.sh all
   ```
   这会运行 pre-commit 检查和 pytest 测试，验证大部分 CI 功能。

2. **推送到 GitHub**（步骤 3）进行最终验证：
   - 创建测试分支并推送
   - 在 GitHub Actions 页面查看运行结果
   - GitHub 会在云端运行 workflows，无需本地 Docker

这种方式已经足够验证 workflows 的正确性。步骤 2（act）主要用于在推送前更完整地模拟 GitHub Actions 环境，但不是必需的。

## 故障排查

### 本地测试问题

**问题**: pytest 找不到测试文件
- **解决**: 确保在项目根目录运行，且 `tests/` 目录存在

**问题**: pre-commit hooks 失败
- **解决**: 运行 `pre-commit run --all-files` 查看具体错误，修复后重试

**问题**: act 无法下载镜像
- **解决**: 检查网络连接，或手动拉取镜像：`docker pull catthehacker/ubuntu:act-latest`
- **如果没有 Docker**: 跳过步骤 2，直接使用步骤 1 和步骤 3

### GitHub Actions 问题

**问题**: workflow 未触发
- **解决**: 检查触发条件（分支名、事件类型）是否正确

**问题**: 权限错误
- **解决**: 检查 workflow 中的 `permissions` 设置是否足够

**问题**: 依赖安装失败
- **解决**: 检查 `pyproject.toml` 中的依赖配置是否正确

## 参考资源

- [GitHub Actions 文档](https://docs.github.com/en/actions)
- [act 工具文档](https://github.com/nektos/act)
- [pre-commit 文档](https://pre-commit.com/)
- [pytest 文档](https://docs.pytest.org/)

