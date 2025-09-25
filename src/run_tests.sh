#!/bin/bash
# QwenImage Generation 测试运行脚本

echo "🎯 QwenImage Generation 测试脚本"
echo "=================================="

# 检查 Python 环境
echo "🔍 检查 Python 环境..."
python3 --version

# 检查必要的包
echo "📦 检查必要的包..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}')"

# 检查 diffusers
if python3 -c "import diffusers" 2>/dev/null; then
    python3 -c "import diffusers; print(f'diffusers: {diffusers.__version__}')"
else
    echo "❌ diffusers 未安装"
    echo "💡 请安装: pip install diffusers"
    exit 1
fi

# 检查 vLLM
if python3 -c "import vllm" 2>/dev/null; then
    python3 -c "import vllm; print(f'vLLM: {vllm.__version__}')"
else
    echo "❌ vLLM 未安装"
    echo "💡 请安装: pip install vllm"
    exit 1
fi

echo ""
echo "🧪 运行基础测试..."
echo "=================="

# 运行基础测试
python3 test_qwen_image_gen.py

echo ""
echo "🧪 运行 vLLM 集成测试..."
echo "========================"

# 运行 vLLM 集成测试
python3 test_vllm_integration.py

echo ""
echo "🎉 测试完成！"
echo "============="
