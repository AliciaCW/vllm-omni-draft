#!/bin/bash
# 使用 act 工具本地运行 GitHub Actions
# 需要先安装 act: https://github.com/nektos/act

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_header() {
    echo -e "\n${GREEN}=== $1 ===${NC}\n"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}" >&2
}

# 检查 act 是否安装
check_act() {
    if ! command -v act &> /dev/null; then
        print_error "act 未安装"
        echo "安装方法:"
        echo "  macOS:   brew install act"
        echo "  Linux:   curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash"
        echo "  或访问:  https://github.com/nektos/act#installation"
        exit 1
    fi
    echo "act 版本: $(act --version)"
}

# 运行指定的 workflow
run_workflow() {
    local workflow=$1
    local event=${2:-push}
    
    print_header "运行 workflow: $workflow (事件: $event)"
    
    # act 命令说明:
    # -W: workflow 文件路径
    # -e: 事件文件（可选，用于模拟 PR 等）
    # --container-architecture: 使用 linux/amd64（避免 M1/M2 Mac 的兼容性问题）
    # -P: 使用自定义 runner 镜像（可选）
    
    case $event in
        push)
            act push -W ".github/workflows/$workflow" --container-architecture linux/amd64
            ;;
        pull_request)
            act pull_request -W ".github/workflows/$workflow" --container-architecture linux/amd64
            ;;
        workflow_dispatch)
            act workflow_dispatch -W ".github/workflows/$workflow" --container-architecture linux/amd64
            ;;
        *)
            act -W ".github/workflows/$workflow" --eventpath <(echo "{\"$event\": {}}") --container-architecture linux/amd64
            ;;
    esac
}

# 主函数
main() {
    local workflow=${1:-all}
    local event=${2:-push}
    
    check_act
    
    case $workflow in
        ci)
            run_workflow "ci.yml" "$event"
            ;;
        pre-commit)
            run_workflow "pre-commit.yml" "$event"
            ;;
        automerge)
            echo -e "${YELLOW}注意: automerge.yml 需要真实的 GitHub token，本地测试可能无法完全模拟${NC}"
            run_workflow "automerge.yml" "pull_request"
            ;;
        all)
            print_header "运行所有 workflows"
            run_workflow "pre-commit.yml" "$event"
            run_workflow "ci.yml" "$event"
            ;;
        *)
            echo "用法: $0 [ci|pre-commit|automerge|all] [push|pull_request|workflow_dispatch]"
            echo ""
            echo "示例:"
            echo "  $0 ci push              # 运行 ci.yml，模拟 push 事件"
            echo "  $0 pre-commit           # 运行 pre-commit.yml，默认 push 事件"
            echo "  $0 all pull_request     # 运行所有 workflows，模拟 PR 事件"
            exit 1
            ;;
    esac
}

main "$@"

