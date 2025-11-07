#!/bin/bash
# 本地模拟 CI 测试脚本
# 使用方法: ./test_ci_local.sh [ci|pre-commit|all]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${GREEN}=== $1 ===${NC}\n"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}" >&2
}

# 检查 Python 版本
check_python() {
    local version=$1
    print_header "检查 Python $version"
    
    if command -v python${version} &> /dev/null; then
        python${version} --version
    elif command -v python${version%.*} &> /dev/null; then
        python${version%.*} --version
    else
        print_error "未找到 Python $version，请先安装"
        return 1
    fi
}

# 模拟 ci.yml 的测试步骤
test_ci() {
    print_header "模拟 CI 测试 (ci.yml)"
    
    # 检查 Python 版本（测试多个版本）
    local versions=("3.9" "3.10" "3.11")
    local python_cmd=""
    
    for version in "${versions[@]}"; do
        if command -v python${version} &> /dev/null || command -v python${version%.*} &> /dev/null; then
            if command -v python${version} &> /dev/null; then
                python_cmd="python${version}"
            else
                python_cmd="python${version%.*}"
            fi
            echo -e "\n${YELLOW}使用 $python_cmd 进行测试${NC}"
            break
        fi
    done
    
    if [ -z "$python_cmd" ]; then
        python_cmd="python3"
        echo -e "${YELLOW}使用默认 $python_cmd${NC}"
    fi
    
    # 升级 pip
    print_header "升级 pip"
    $python_cmd -m pip install --upgrade pip
    
    # 安装依赖
    print_header "安装依赖 (dev)"
    $python_cmd -m pip install -e ".[dev]"
    
    # 运行测试
    print_header "运行 pytest (排除 slow 标记)"
    $python_cmd -m pytest tests/ -v -m "not slow"
    
    echo -e "\n${GREEN}✓ CI 测试完成${NC}"
}

# 模拟 pre-commit.yml 的检查步骤
test_precommit() {
    print_header "模拟 pre-commit 检查 (pre-commit.yml)"
    
    # 检查 pre-commit 是否安装
    if ! command -v pre-commit &> /dev/null; then
        print_header "安装 pre-commit"
        pip install pre-commit
    fi
    
    # 安装 pre-commit hooks
    print_header "安装 pre-commit hooks"
    pre-commit install
    
    # 运行 pre-commit（模拟 GitHub Actions 的行为）
    print_header "运行 pre-commit (--all-files --hook-stage manual)"
    pre-commit run --all-files --hook-stage manual
    
    echo -e "\n${GREEN}✓ pre-commit 检查完成${NC}"
}

# 主函数
main() {
    local test_type=${1:-all}
    
    case $test_type in
        ci)
            test_ci
            ;;
        pre-commit)
            test_precommit
            ;;
        all)
            test_precommit
            test_ci
            ;;
        *)
            echo "用法: $0 [ci|pre-commit|all]"
            echo "  ci         - 只运行 CI 测试"
            echo "  pre-commit  - 只运行 pre-commit 检查"
            echo "  all        - 运行所有测试（默认）"
            exit 1
            ;;
    esac
}

main "$@"

