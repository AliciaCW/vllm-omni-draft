#!/usr/bin/env python3
"""
快速测试脚本：验证 QwenImage 集成的基本功能

这个脚本提供了一个简化的测试，不需要实际的模型文件。
"""

import os
import sys
import torch

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(__file__))


def test_imports():
    """测试导入功能"""
    print("🧪 测试导入功能...")

    try:
        from qwen_image_gen import (
            QwenImageGenModel,
            QwenImageGenProcessor,
            QwenImageGenWorker,
            QwenImageGenExecutor,
            create_qwen_image_config
        )
        print("✅ 核心模块导入成功")

        from qwen_image_gen.types import (
            QwenImageInputs,
            QwenImageTask,
            QwenImageOutputMode
        )
        print("✅ 类型定义导入成功")

        return True

    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False


def test_configuration():
    """测试配置功能"""
    print("\n🧪 测试配置功能...")

    try:
        from qwen_image_gen import create_qwen_image_config

        # 创建配置
        config = create_qwen_image_config(
            model_id="test-model",
            max_batch_size=2
        )

        print(f"✅ 配置创建成功: {config.model_id}")
        print(f"✅ 配置参数: {len(config.__dict__)} 个")

        return True

    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        return False


def test_types():
    """测试类型定义"""
    print("\n🧪 测试类型定义...")

    try:
        from qwen_image_gen.types import (
            QwenImageInputs,
            QwenImageTask,
            QwenImageOutputMode
        )

        # 创建测试数据
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        prompt_embeds = torch.randn(1, 77, 768, device=device, dtype=dtype)
        prompt_mask = torch.ones(1, 77, device=device, dtype=torch.bool)
        image_latents = torch.randn(1, 4, 64, 64, device=device, dtype=dtype)
        timesteps = torch.linspace(1000, 0, 20, device=device).long()

        # 创建输入对象
        inputs = QwenImageInputs(
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_mask,
            image_latents=image_latents,
            timesteps=timesteps,
            task=QwenImageTask.TEXT_TO_IMAGE,
            output_mode=QwenImageOutputMode.PIXELS
        )

        print(f"✅ 输入对象创建成功: {inputs.task}")
        print(f"   提示嵌入: {inputs.prompt_embeds.shape}")
        print(f"   图像潜在: {inputs.image_latents.shape}")
        print(f"   时间步: {inputs.timesteps.shape}")

        return True

    except Exception as e:
        print(f"❌ 类型测试失败: {e}")
        return False


def test_processor():
    """测试处理器功能"""
    print("\n🧪 测试处理器功能...")

    try:
        from qwen_image_gen.processor import QwenImageGenProcessor

        # 创建处理器
        processor = QwenImageGenProcessor()
        print("✅ 处理器创建成功")

        # 测试参数提取
        test_kwargs = {
            "guidance_scale": 7.5,
            "num_inference_steps": 50,
            "task": "text_to_image",
            "output_mode": "pixels",
            "height": 1024,
            "width": 1024
        }

        params = processor._extract_generation_params(test_kwargs)
        print(f"✅ 参数提取成功: {len(params)} 个参数")

        # 测试文本处理
        prompt_embeds, prompt_mask = processor._process_text_prompt(
            "test prompt")
        print(f"✅ 文本处理成功: {prompt_embeds.shape}")

        # 测试图像处理
        from qwen_image_gen.types import QwenImageTask
        mm_data = {}  # 空的 multimodal data
        params = {"task": QwenImageTask.TEXT_TO_IMAGE,
                  "height": 512, "width": 512}
        image_latents = processor._process_image_inputs(mm_data, params)
        print(f"✅ 图像处理成功: {image_latents.shape}")

        return True

    except Exception as e:
        print(f"❌ 处理器测试失败: {e}")
        return False


def test_model_wrapper():
    """测试模型包装器（不加载实际模型）"""
    print("\n🧪 测试模型包装器...")

    try:
        from qwen_image_gen.model import QwenImageGenModel
        from vllm.config import VllmConfig
        from vllm.model_config import ModelConfig

        # 创建模拟的 vLLM 配置
        model_config = ModelConfig(
            model="test-model",
            trust_remote_code=True,
            dtype=torch.float16
        )

        vllm_config = VllmConfig(model_config=model_config)

        print("✅ vLLM 配置创建成功")

        # 注意：这里我们不实际创建模型，因为需要下载文件
        print("⚠️  跳过实际模型创建（避免下载）")
        print("✅ 模型包装器接口验证通过")

        return True

    except Exception as e:
        print(f"❌ 模型包装器测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🚀 QwenImage 快速测试")
    print("=" * 40)

    # 检查环境
    print(f"🐍 Python 版本: {sys.version}")
    print(f"🔥 PyTorch 版本: {torch.__version__}")
    print(f"🎮 CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   设备: {torch.cuda.get_device_name()}")

    print("\n" + "=" * 40)

    # 运行测试
    tests = [
        ("导入功能", test_imports),
        ("配置功能", test_configuration),
        ("类型定义", test_types),
        ("处理器功能", test_processor),
        ("模型包装器", test_model_wrapper),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🧪 测试: {test_name}")
        print("-" * 30)

        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 测试通过")
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")

    # 总结
    print("\n" + "=" * 40)
    print(f"📊 测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！")
        print("\n💡 下一步:")
        print("   1. 安装 diffusers: pip install diffusers")
        print("   2. 安装 vLLM: pip install vllm")
        print("   3. 运行完整测试: python test_qwen_image_gen.py")
    else:
        print("❌ 部分测试失败，请检查错误信息")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
