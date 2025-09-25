# QwenImage Generation 测试指南

这个文档提供了如何使用和测试 QwenImage 生成集成的详细指南。

## 📋 测试脚本概览

我们提供了三个不同层次的测试脚本：

### 1. `quick_test.py` - 快速验证测试
- **用途**: 验证基本功能和导入
- **特点**: 不需要实际模型文件，运行快速
- **适用**: 开发阶段的基本验证

### 2. `test_qwen_image_gen.py` - 完整功能测试
- **用途**: 测试所有组件的完整功能
- **特点**: 模拟完整的生成流程
- **适用**: 功能验证和调试

### 3. `test_vllm_integration.py` - vLLM 集成测试
- **用途**: 测试与 vLLM 的实际集成
- **特点**: 需要实际的模型文件和 vLLM 环境
- **适用**: 生产环境验证

## 🚀 快速开始

### 步骤 1: 环境准备

```bash
# 安装基础依赖
pip install torch torchvision

# 安装 diffusers（用于 QwenImage 模型）
pip install diffusers

# 安装 vLLM（用于推理引擎）
pip install vllm
```

### 步骤 2: 运行快速测试

```bash
# 运行快速验证测试
python quick_test.py
```

这个测试会验证：
- ✅ 模块导入
- ✅ 配置系统
- ✅ 类型定义
- ✅ 处理器功能
- ✅ 模型包装器接口

### 步骤 3: 运行完整测试

```bash
# 运行完整功能测试
python test_qwen_image_gen.py
```

这个测试会验证：
- ✅ 环境配置
- ✅ 输入处理
- ✅ 生成流程模拟
- ✅ 结果保存

### 步骤 4: 运行 vLLM 集成测试

```bash
# 运行 vLLM 集成测试
python test_vllm_integration.py
```

**注意**: 这个测试需要实际的模型文件，可能会下载大文件。

## 🔧 环境配置

### 环境变量

您可以通过环境变量来配置测试：

```bash
# 模型配置
export QWEN_TRANSFORMER_MODEL_ID="Qwen/Qwen-Image"
export QWEN_VAE_MODEL_ID="Qwen/Qwen-Image"
export QWEN_TRANSFORMER_SUBFOLDER="transformer"
export QWEN_VAE_SUBFOLDER="vae"

# 生成参数
export QWEN_HEIGHT="512"
export QWEN_WIDTH="512"
export QWEN_GUIDANCE_SCALE="4.0"
export QWEN_NUM_STEPS="20"

# 性能配置
export QWEN_MAX_BATCH_SIZE="2"
export QWEN_DEBUG="1"
```

### 硬件要求

- **GPU**: 推荐使用 NVIDIA GPU，至少 8GB 显存
- **内存**: 至少 16GB 系统内存
- **存储**: 至少 10GB 可用空间（用于模型文件）

## 📊 测试结果解读

### 快速测试结果

```
🚀 QwenImage 快速测试
========================================
🐍 Python 版本: 3.9.0
🔥 PyTorch 版本: 2.0.0
🎮 CUDA 可用: True
   设备: NVIDIA GeForce RTX 4090

========================================

🧪 测试: 导入功能
------------------------------
✅ 核心模块导入成功
✅ 类型定义导入成功
✅ 导入功能 测试通过

🧪 测试: 配置功能
------------------------------
✅ 配置创建成功: test-model
✅ vLLM 配置转换成功: 12 个参数
✅ 配置验证通过
✅ 配置功能 测试通过

📊 测试结果: 5/5 通过
🎉 所有测试通过！
```

### 完整测试结果

```
🚀 开始 QwenImage Generation 综合测试
============================================================
🔧 设置环境配置...
✅ 环境配置完成

🧪 测试配置系统...
✅ 环境变量配置:
   - Transformer 模型: Qwen/Qwen-Image
   - VAE 模型: Qwen/Qwen-Image
   - Transformer 子文件夹: transformer
   - VAE 子文件夹: vae
   - 最大批次大小: 2
   - 默认尺寸: 512x512
   - 引导比例: 4.0
   - 推理步数: 20
   - 设备: cuda
   - 数据类型: torch.float16

✅ vLLM 配置转换:
   - model: Qwen/Qwen-Image
   - trust_remote_code: True
   - dtype: torch.float16
   - gpu_memory_utilization: 0.8
   - max_num_seqs: 2
   - worker_cls: qwen_image_gen.worker.QwenImageGenWorker

✅ 配置验证通过

🧪 测试输入处理...
📝 创建模拟文本嵌入: batch_size=1, seq_len=77, embed_dim=768
✅ 文本嵌入创建完成: torch.Size([1, 77, 768])
🎲 创建初始噪声: batch_size=1, height=512, width=512
✅ 初始噪声创建完成: torch.Size([1, 4, 64, 64])
⏰ 创建时间步: num_steps=20, batch_size=1
✅ 时间步创建完成: torch.Size([1, 20])
✅ 输入处理测试成功
   - 提示嵌入: torch.Size([1, 77, 768])
   - 图像潜在: torch.Size([1, 4, 64, 64])
   - 时间步: torch.Size([1, 20])
   - 任务: QwenImageTask.TEXT_TO_IMAGE
   - 输出模式: QwenImageOutputMode.PIXELS

🧪 测试简单生成...
🎨 模拟图像生成过程...
   初始潜在表示: torch.Size([1, 4, 64, 64])
   步骤 1/20: 模拟去噪...
     潜在表示范围: [-0.123, 0.456]
   步骤 2/20: 模拟去噪...
     潜在表示范围: [-0.098, 0.423]
   ...
🖼️  模拟 VAE 解码...
   解码像素图像: torch.Size([1, 3, 512, 512])
   像素值范围: [0.000, 1.000]
✅ 生成完成: torch.Size([1, 3, 512, 512])

💾 保存测试图像: test_generated_image.png
✅ 图像保存成功: test_generated_image.png
   图像尺寸: (512, 512)

============================================================
🎉 综合测试完成！

📋 测试总结:
   ✅ 环境配置
   ✅ 配置系统
   ✅ 输入处理
   ✅ 生成流程
   ✅ 结果保存

💡 下一步:
   1. 确保有实际的 QwenImage 模型文件
   2. 安装 diffusers: pip install diffusers
   3. 运行完整的 vLLM 集成测试
```

## 🐛 常见问题排查

### 1. 导入错误

**问题**: `ModuleNotFoundError: No module named 'qwen_image_gen'`

**解决方案**:
```bash
# 确保在正确的目录中运行
cd /path/to/vllm-omni-qwen-image

# 检查 Python 路径
python -c "import sys; print(sys.path)"
```

### 2. CUDA 错误

**问题**: `CUDA out of memory`

**解决方案**:
```bash
# 减少批次大小
export QWEN_MAX_BATCH_SIZE="1"

# 减少图像尺寸
export QWEN_HEIGHT="256"
export QWEN_WIDTH="256"

# 减少推理步数
export QWEN_NUM_STEPS="10"
```

### 3. 模型下载错误

**问题**: `OSError: [Errno 2] No such file or directory`

**解决方案**:
```bash
# 检查网络连接
ping huggingface.co

# 手动下载模型
git lfs install
git clone https://huggingface.co/Qwen/Qwen-Image
```

### 4. vLLM 版本兼容性

**问题**: `AttributeError: 'VllmConfig' object has no attribute 'xxx'`

**解决方案**:
```bash
# 检查 vLLM 版本
pip show vllm

# 升级到最新版本
pip install --upgrade vllm
```

## 📈 性能优化建议

### 1. 内存优化

```bash
# 启用内存高效注意力
export QWEN_MEMORY_EFFICIENT_ATTENTION="1"

# 调整 GPU 内存使用
export QWEN_GPU_MEMORY_UTIL="0.7"

# 启用 CPU 卸载
export QWEN_CPU_OFFLOAD_GB="2.0"
```

### 2. 速度优化

```bash
# 启用 torch 编译
export QWEN_TORCH_COMPILE="1"

# 减少推理步数
export QWEN_NUM_STEPS="20"

# 使用较小的图像尺寸
export QWEN_HEIGHT="256"
export QWEN_WIDTH="256"
```

### 3. 批处理优化

```bash
# 根据 GPU 内存调整批次大小
export QWEN_MAX_BATCH_SIZE="4"  # 对于 24GB GPU
export QWEN_MAX_BATCH_SIZE="2"  # 对于 12GB GPU
export QWEN_MAX_BATCH_SIZE="1"  # 对于 8GB GPU
```

## 🔍 调试技巧

### 1. 启用调试模式

```bash
export QWEN_DEBUG="1"
export QWEN_LOG_LEVEL="DEBUG"
```

### 2. 检查中间结果

在测试脚本中添加打印语句：

```python
# 在生成过程中添加调试信息
print(f"潜在表示形状: {latents.shape}")
print(f"潜在表示范围: [{latents.min():.3f}, {latents.max():.3f}]")
```

### 3. 保存中间结果

```python
# 保存中间潜在表示
torch.save(latents, "intermediate_latents.pt")

# 保存生成的图像
Image.fromarray(pixels).save("debug_image.png")
```

## 📚 进一步学习

- [Qwen-Image 官方仓库](https://huggingface.co/Qwen/Qwen-Image)
- [QwenImage 官方文档](https://github.com/QwenLM/Qwen-Image)
- [vLLM 官方文档](https://docs.vllm.ai/)
- [Diffusers 官方文档](https://huggingface.co/docs/diffusers)

## 🤝 贡献

如果您发现问题或有改进建议，请：

1. 提交 Issue 描述问题
2. 提供完整的错误日志
3. 说明您的环境配置
4. 提供复现步骤
