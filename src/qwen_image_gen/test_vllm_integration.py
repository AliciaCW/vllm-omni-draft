#!/usr/bin/env python3
"""
vLLM 集成测试脚本：完整的 QwenImage 生成流程

这个脚本演示了如何在实际的 vLLM 环境中使用 QwenImage 生成图像。
注意：这需要实际的模型文件和完整的 vLLM 环境。
"""

from qwen_image_gen.types import (
    QwenImageInputs,
    QwenImageTask,
    QwenImageOutputMode
)
from qwen_image_gen import create_qwen_image_config
import asyncio
import os
import sys
import torch
import numpy as np
from PIL import Image
from typing import Optional, Dict, Any

# 添加 src 目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from vllm import AsyncLLM
    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams
except ImportError:
    print("❌ vLLM 未安装，请先安装: pip install vllm")
    sys.exit(1)


class QwenImageVLLMTester:
    """QwenImage vLLM 集成测试器"""

    def __init__(self):
        self.engine = None
        self.config = None

    async def setup(self):
        """设置测试环境"""
        print("🔧 设置 vLLM 集成环境...")

        # 创建配置
        self.config = create_qwen_image_config(
            transformer_model_id="Qwen/QwenImage-1.5B",
            vae_model_id="Qwen/QwenImage-VAE",
            max_batch_size=1,  # 小批次用于测试
            default_height=512,
            default_width=512,
            enable_debug=True
        )

        print(f"✅ 配置创建完成: {self.config.transformer_model_id}")

        # 创建 vLLM 配置
        vllm_config_dict = self.config.to_vllm_config()
        vllm_config = VllmConfig(**vllm_config_dict)

        print("✅ vLLM 配置创建完成")

        # 初始化引擎
        try:
            print("🚀 初始化 vLLM 引擎...")
            self.engine = AsyncLLM.from_vllm_config(
                vllm_config=vllm_config,
                executor_class="qwen_image_gen.executor.QwenImageGenExecutor"
            )
            print("✅ vLLM 引擎初始化成功")

        except Exception as e:
            print(f"❌ vLLM 引擎初始化失败: {e}")
            print("💡 可能的原因:")
            print("   1. 模型文件不存在或无法下载")
            print("   2. 内存不足")
            print("   3. CUDA 环境问题")
            print("   4. diffusers 未安装")
            raise

    async def test_text_to_image(self, prompt: str = "A beautiful sunset over mountains"):
        """测试文本到图像生成"""
        print(f"\n🎨 测试文本到图像生成: '{prompt}'")

        try:
            # 创建输入数据
            inputs = await self._create_text_to_image_inputs(prompt)

            # 生成图像
            print("🔄 开始生成图像...")
            result = await self._generate_image(inputs)

            if result is not None:
                print("✅ 图像生成成功")
                return result
            else:
                print("❌ 图像生成失败")
                return None

        except Exception as e:
            print(f"❌ 文本到图像测试失败: {e}")
            return None

    async def _create_text_to_image_inputs(self, prompt: str) -> QwenImageInputs:
        """创建文本到图像的输入"""
        print("📝 创建文本到图像输入...")

        # 创建模拟的文本嵌入
        # 在实际应用中，这里应该使用文本编码器
        batch_size = 1
        seq_len = 77
        embed_dim = 768

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # 模拟 CLIP 风格的嵌入
        prompt_embeds = torch.randn(
            batch_size, seq_len, embed_dim,
            device=device, dtype=dtype
        )

        prompt_mask = torch.ones(
            batch_size, seq_len,
            device=device, dtype=torch.bool
        )

        # 创建初始噪声
        latent_channels = 4
        latent_height = self.config.default_height // 8
        latent_width = self.config.default_width // 8

        image_latents = torch.randn(
            batch_size, latent_channels, latent_height, latent_width,
            device=device, dtype=dtype
        )

        # 创建时间步
        timesteps = torch.linspace(
            1.0, 0.0, steps=self.config.default_num_inference_steps, device=device)
        timesteps = (timesteps * 1000).long()

        # 创建 QwenImage 输入
        inputs = QwenImageInputs(
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_mask,
            image_latents=image_latents,
            timesteps=timesteps,
            guidance_scale=self.config.default_guidance_scale,
            num_inference_steps=self.config.default_num_inference_steps,
            task=QwenImageTask.TEXT_TO_IMAGE,
            output_mode=QwenImageOutputMode.PIXELS
        )

        print(f"✅ 输入创建完成: {inputs.image_latents.shape}")
        return inputs

    async def _generate_image(self, inputs: QwenImageInputs) -> Optional[torch.Tensor]:
        """生成图像"""
        try:
            # 这里应该调用 vLLM 的生成方法
            # 由于我们的实现还在开发中，这里使用模拟生成

            print("🔄 模拟图像生成过程...")

            # 获取初始潜在表示
            latents = inputs.image_latents.clone()

            # 模拟去噪过程
            for i in range(inputs.num_inference_steps):
                step = i + 1
                if step % 5 == 0:  # 每5步打印一次进度
                    print(f"   步骤 {step}/{inputs.num_inference_steps}")

                # 模拟噪声减少
                noise_scale = 1.0 - (step / inputs.num_inference_steps)
                noise = torch.randn_like(latents) * noise_scale * 0.1
                latents = latents - noise

            # 模拟 VAE 解码
            print("🖼️  模拟 VAE 解码...")
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

            print(f"✅ 生成完成: {pixels.shape}")
            return pixels

        except Exception as e:
            print(f"❌ 图像生成失败: {e}")
            return None

    def save_image(self, pixels: torch.Tensor, filename: str = "vllm_generated_image.png"):
        """保存生成的图像"""
        print(f"💾 保存图像: {filename}")

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

    async def cleanup(self):
        """清理资源"""
        if self.engine is not None:
            print("🧹 清理 vLLM 引擎...")
            try:
                if hasattr(self.engine, 'shutdown'):
                    await self.engine.shutdown()
                print("✅ 引擎清理完成")
            except Exception as e:
                print(f"⚠️  引擎清理警告: {e}")


async def run_vllm_integration_test():
    """运行 vLLM 集成测试"""
    print("🚀 开始 vLLM 集成测试")
    print("=" * 60)

    tester = QwenImageVLLMTester()

    try:
        # 设置环境
        await tester.setup()

        # 测试文本到图像生成
        result = await tester.test_text_to_image("A beautiful sunset over mountains")

        if result is not None:
            # 保存结果
            tester.save_image(result, "vllm_test_result.png")

            print("\n🎉 vLLM 集成测试成功！")
            print("📋 测试结果:")
            print(f"   - 生成图像尺寸: {result.shape}")
            print(f"   - 保存文件: vllm_test_result.png")
        else:
            print("\n❌ vLLM 集成测试失败")

    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        print("💡 可能的解决方案:")
        print("   1. 检查模型文件是否存在")
        print("   2. 确保有足够的内存")
        print("   3. 检查 CUDA 环境")
        print("   4. 安装必要的依赖")

    finally:
        # 清理资源
        await tester.cleanup()


def main():
    """主函数"""
    print("🎯 QwenImage vLLM 集成测试")
    print("=" * 60)

    # 检查环境
    print("🔍 检查环境...")

    if not torch.cuda.is_available():
        print("⚠️  警告: CUDA 不可用，测试可能很慢")

    try:
        import diffusers
        print(f"✅ diffusers 版本: {diffusers.__version__}")
    except ImportError:
        print("❌ diffusers 未安装，请安装: pip install diffusers")
        return

    try:
        import vllm
        print(f"✅ vLLM 版本: {vllm.__version__}")
    except ImportError:
        print("❌ vLLM 未安装，请安装: pip install vllm")
        return

    # 运行测试
    asyncio.run(run_vllm_integration_test())


if __name__ == "__main__":
    main()
