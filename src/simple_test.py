#!/usr/bin/env python3
"""
简化测试脚本：验证 QwenImage 集成的核心功能
"""

import os
import sys
import torch

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(__file__))


def test_basic_imports():
    """测试基本导入功能"""
    print("🧪 测试基本导入...")

    try:
        from qwen_image_gen import create_qwen_image_config
        print("✅ 配置导入成功")

        from qwen_image_gen.types import QwenImageInputs, QwenImageTask, QwenImageOutputMode
        print("✅ 类型导入成功")

        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False


def test_config_creation():
    """测试配置创建"""
    print("\n🧪 测试配置创建...")

    try:
        from qwen_image_gen import create_qwen_image_config

        config = create_qwen_image_config(
            model_id="test-model",
            max_batch_size=2
        )

        print(f"✅ 配置创建成功: {config.model_id}")
        print(f"   - 批次大小: {config.max_batch_size}")
        print(f"   - 设备: {config.device}")
        print(f"   - 数据类型: {config.dtype}")

        return True
    except Exception as e:
        print(f"❌ 配置创建失败: {e}")
        return False


def test_types_creation():
    """测试类型创建"""
    print("\n🧪 测试类型创建...")

    try:
        from qwen_image_gen.types import QwenImageInputs, QwenImageTask, QwenImageOutputMode

        # 创建测试数据
        batch_size = 1
        seq_len = 77
        embed_dim = 768
        latent_channels = 4
        latent_height = 64
        latent_width = 64

        prompt_embeds = torch.randn(batch_size, seq_len, embed_dim)
        prompt_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        image_latents = torch.randn(
            batch_size, latent_channels, latent_height, latent_width)
        timesteps = torch.linspace(1.0, 0.0, steps=10).long()

        # 创建输入对象
        qwen_inputs = QwenImageInputs(
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_mask,
            image_latents=image_latents,
            timesteps=timesteps,
            task=QwenImageTask.TEXT_TO_IMAGE,
            output_mode=QwenImageOutputMode.PIXELS
        )

        print(f"✅ 输入对象创建成功")
        print(f"   - 任务类型: {qwen_inputs.task}")
        print(f"   - 输出模式: {qwen_inputs.output_mode}")
        print(f"   - 引导比例: {qwen_inputs.guidance_scale}")
        print(f"   - 推理步数: {qwen_inputs.num_inference_steps}")

        return True
    except Exception as e:
        print(f"❌ 类型创建失败: {e}")
        return False


def test_processor_basic():
    """测试处理器基本功能"""
    print("\n🧪 测试处理器基本功能...")

    try:
        from qwen_image_gen.processor import QwenImageGenProcessor

        # 创建处理器
        processor = QwenImageGenProcessor()
        print("✅ 处理器创建成功")

        # 测试参数提取
        test_kwargs = {
            "guidance_scale": 5.0,
            "num_inference_steps": 20,
            "height": 256,
            "width": 256
        }

        params = processor._extract_generation_params(test_kwargs)
        print(f"✅ 参数提取成功: {len(params)} 个参数")

        return True
    except Exception as e:
        print(f"❌ 处理器测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🚀 开始简化测试...")

    tests = [
        test_basic_imports,
        test_config_creation,
        test_types_creation,
        test_processor_basic,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print(f"\n📊 测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！")
        return True
    else:
        print("❌ 部分测试失败")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
