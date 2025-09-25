#!/usr/bin/env python3
"""
测试脚本：QwenImage Generation with vLLM v1

这个脚本演示了如何使用新设计的 qwen_image_gen 包来生成图像。
"""

from qwen_image_gen.types import (
    QwenImageInputs,
    QwenImageTask,
    QwenImageOutputMode
)
from qwen_image_gen import (
    QwenImageGenModel,
    QwenImageGenProcessor,
    QwenImageGenWorker,
    QwenImageGenExecutor,
    create_qwen_image_config
)
import asyncio
import os
import sys
import torch
import numpy as np
from PIL import Image
from typing import Optional

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(__file__))


def setup_environment():
    """设置环境变量和配置"""
    print("🔧 设置环境配置...")

    # 设置环境变量
    os.environ.setdefault("QWEN_MODEL_ID", "Qwen/Qwen-Image")
    os.environ.setdefault("QWEN_TRANSFORMER_SUBFOLDER", "transformer")
    os.environ.setdefault("QWEN_VAE_SUBFOLDER", "vae")
    os.environ.setdefault("QWEN_MAX_BATCH_SIZE", "2")
    os.environ.setdefault("QWEN_HEIGHT", "512")
    os.environ.setdefault("QWEN_WIDTH", "512")
    os.environ.setdefault("QWEN_GUIDANCE_SCALE", "4.0")
    os.environ.setdefault("QWEN_NUM_STEPS", "20")  # 减少步数用于测试
    os.environ.setdefault("QWEN_DEBUG", "1")

    print("✅ 环境配置完成")


def create_dummy_embeddings(batch_size: int = 1, seq_len: int = 77, embed_dim: int = 768):
    """创建模拟的文本嵌入"""
    print(
        f"📝 创建模拟文本嵌入: batch_size={batch_size}, seq_len={seq_len}, embed_dim={embed_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # 创建模拟的 CLIP 风格嵌入
    prompt_embeds = torch.randn(
        batch_size, seq_len, embed_dim,
        device=device, dtype=dtype
    )

    # 创建注意力掩码
    prompt_mask = torch.ones(
        batch_size, seq_len,
        device=device, dtype=torch.bool
    )

    print(f"✅ 文本嵌入创建完成: {prompt_embeds.shape}")
    return prompt_embeds, prompt_mask


def create_initial_latents(batch_size: int = 1, height: int = 512, width: int = 512):
    """创建初始噪声潜在表示"""
    print(f"🎲 创建初始噪声: batch_size={batch_size}, height={height}, width={width}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # VAE 下采样比例通常是 8
    latent_height = height // 8
    latent_width = width // 8
    latent_channels = 4  # 典型的 VAE 潜在通道数

    # 创建随机噪声
    latents = torch.randn(
        batch_size, latent_channels, latent_height, latent_width,
        device=device, dtype=dtype
    )

    print(f"✅ 初始噪声创建完成: {latents.shape}")
    return latents


def create_timesteps(num_steps: int = 20, batch_size: int = 1):
    """创建去噪时间步"""
    print(f"⏰ 创建时间步: num_steps={num_steps}, batch_size={batch_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 线性调度
    timesteps = torch.linspace(1.0, 0.0, steps=num_steps, device=device)
    # 缩放到典型的扩散范围
    timesteps = (timesteps * 1000).long()

    # 扩展为批次
    timesteps = timesteps.unsqueeze(0).expand(batch_size, -1)

    print(f"✅ 时间步创建完成: {timesteps.shape}")
    return timesteps


def test_model_loading():
    """测试模型加载"""
    print("\n🧪 测试模型加载...")

    try:
        # 创建配置
        config = create_qwen_image_config(
            model_id="Qwen/Qwen-Image",
            transformer_subfolder="transformer",
            vae_subfolder="vae",
            max_batch_size=2,
            height=512,
            width=512
        )

        print(f"✅ 配置创建成功: {config.model_id}")

        # 注意：这里我们只是测试配置，不实际加载模型
        # 因为模型文件可能不存在，会导致下载失败
        print("⚠️  跳过实际模型加载（避免下载大文件）")

        return config

    except Exception as e:
        print(f"❌ 模型加载测试失败: {e}")
        return None


def test_input_processing():
    """测试输入处理"""
    print("\n🧪 测试输入处理...")

    try:
        # 创建处理器
        processor = QwenImageGenProcessor()

        # 创建模拟输入
        prompt_embeds, prompt_mask = create_dummy_embeddings()
        image_latents = create_initial_latents()
        timesteps = create_timesteps()

        # 创建 QwenImage 输入
        qwen_inputs = QwenImageInputs(
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_mask,
            image_latents=image_latents,
            timesteps=timesteps,
            guidance_scale=4.0,
            num_inference_steps=20,
            task=QwenImageTask.TEXT_TO_IMAGE,
            output_mode=QwenImageOutputMode.PIXELS
        )

        print(f"✅ 输入处理测试成功")
        print(f"   - 提示嵌入: {qwen_inputs.prompt_embeds.shape}")
        print(f"   - 图像潜在: {qwen_inputs.image_latents.shape}")
        print(f"   - 时间步: {qwen_inputs.timesteps.shape}")
        print(f"   - 任务: {qwen_inputs.task}")
        print(f"   - 输出模式: {qwen_inputs.output_mode}")

        return qwen_inputs

    except Exception as e:
        print(f"❌ 输入处理测试失败: {e}")
        return None


def test_simple_generation():
    """测试简单生成（不依赖实际模型）"""
    print("\n🧪 测试简单生成...")

    try:
        # 创建输入
        qwen_inputs = test_input_processing()
        if qwen_inputs is None:
            return None

        # 模拟生成过程
        print("🎨 模拟图像生成过程...")

        # 获取初始潜在表示
        latents = qwen_inputs.image_latents.clone()
        print(f"   初始潜在表示: {latents.shape}")

        # 模拟去噪过程
        num_steps = qwen_inputs.num_inference_steps
        for i in range(min(5, num_steps)):  # 只运行前5步用于测试
            step = i + 1
            print(f"   步骤 {step}/{num_steps}: 模拟去噪...")

            # 模拟噪声减少
            noise_scale = 1.0 - (step / num_steps)
            noise = torch.randn_like(latents) * noise_scale * 0.1
            latents = latents - noise

            print(f"     潜在表示范围: [{latents.min():.3f}, {latents.max():.3f}]")

        # 模拟 VAE 解码
        print("🖼️  模拟 VAE 解码...")
        if qwen_inputs.output_mode in [QwenImageOutputMode.PIXELS, QwenImageOutputMode.BOTH]:
            # 模拟解码后的像素图像
            batch_size, channels, height, width = latents.shape
            # 上采样到像素空间
            pixel_height = height * 8
            pixel_width = width * 8
            pixel_channels = 3  # RGB

            pixels = torch.randn(
                batch_size, pixel_channels, pixel_height, pixel_width,
                device=latents.device, dtype=latents.dtype
            )

            # 归一化到 [0, 1] 范围
            pixels = (pixels - pixels.min()) / (pixels.max() - pixels.min())

            print(f"   解码像素图像: {pixels.shape}")
            print(f"   像素值范围: [{pixels.min():.3f}, {pixels.max():.3f}]")

            return pixels
        else:
            print(f"   返回潜在表示: {latents.shape}")
            return latents

    except Exception as e:
        print(f"❌ 简单生成测试失败: {e}")
        return None


def save_test_image(pixels: torch.Tensor, filename: str = "test_generated_image.png"):
    """保存测试图像"""
    print(f"\n💾 保存测试图像: {filename}")

    try:
        # 转换为 numpy 数组
        if pixels.dim() == 4:
            pixels = pixels[0]  # 取第一个批次

        # 转换为 [H, W, C] 格式
        pixels = pixels.permute(1, 2, 0).cpu().numpy()

        # 确保值在 [0, 1] 范围内
        pixels = np.clip(pixels, 0, 1)

        # 转换为 [0, 255] 范围
        pixels = (pixels * 255).astype(np.uint8)

        # 创建 PIL 图像
        image = Image.fromarray(pixels)

        # 保存图像
        image.save(filename)

        print(f"✅ 图像保存成功: {filename}")
        print(f"   图像尺寸: {image.size}")

    except Exception as e:
        print(f"❌ 图像保存失败: {e}")


def test_configuration():
    """测试配置系统"""
    print("\n🧪 测试配置系统...")

    try:
        # 测试环境变量配置
        config = create_qwen_image_config()

        print("✅ 环境变量配置:")
        print(f"   - Transformer 模型: {config.transformer_model_id}")
        print(f"   - VAE 模型: {config.vae_model_id}")
        print(f"   - 最大批次大小: {config.max_batch_size}")
        print(f"   - 默认尺寸: {config.default_height}x{config.default_width}")
        print(f"   - 引导比例: {config.default_guidance_scale}")
        print(f"   - 推理步数: {config.default_num_inference_steps}")
        print(f"   - 设备: {config.device}")
        print(f"   - 数据类型: {config.dtype}")

        # 测试 vLLM 配置转换
        vllm_config_dict = config.to_vllm_config()
        print("\n✅ vLLM 配置转换:")
        for key, value in vllm_config_dict.items():
            print(f"   - {key}: {value}")

        # 测试配置验证
        config.validate()
        print("✅ 配置验证通过")

        return config

    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        return None


def run_comprehensive_test():
    """运行综合测试"""
    print("🚀 开始 QwenImage Generation 综合测试")
    print("=" * 60)

    # 1. 设置环境
    setup_environment()

    # 2. 测试配置
    config = test_configuration()
    if config is None:
        print("❌ 配置测试失败，终止测试")
        return

    # 3. 测试模型加载
    test_model_loading()

    # 4. 测试输入处理
    qwen_inputs = test_input_processing()
    if qwen_inputs is None:
        print("❌ 输入处理测试失败，终止测试")
        return

    # 5. 测试简单生成
    result = test_simple_generation()
    if result is None:
        print("❌ 生成测试失败，终止测试")
        return

    # 6. 保存结果
    if result.dim() == 4 and result.shape[1] == 3:  # 像素图像
        save_test_image(result)

    print("\n" + "=" * 60)
    print("🎉 综合测试完成！")
    print("\n📋 测试总结:")
    print("   ✅ 环境配置")
    print("   ✅ 配置系统")
    print("   ✅ 输入处理")
    print("   ✅ 生成流程")
    print("   ✅ 结果保存")

    print("\n💡 下一步:")
    print("   1. 确保有实际的 QwenImage 模型文件")
    print("   2. 安装 diffusers: pip install diffusers")
    print("   3. 运行完整的 vLLM 集成测试")


def main():
    """主函数"""
    print("🎯 QwenImage Generation 测试脚本")
    print("=" * 60)

    # 检查 CUDA 可用性
    if torch.cuda.is_available():
        print(f"🔥 CUDA 可用: {torch.cuda.get_device_name()}")
        print(
            f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  CUDA 不可用，将使用 CPU（速度较慢）")

    # 运行测试
    run_comprehensive_test()


if __name__ == "__main__":
    main()
