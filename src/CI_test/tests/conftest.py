"""Pytest fixtures 和测试配置。"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests.utils import create_test_image, create_test_video_array


@pytest.fixture
def rng():
    """随机数生成器 fixture。"""
    return np.random.RandomState(42)


@pytest.fixture
def sample_image(rng):
    """生成示例测试图像。"""
    return create_test_image(width=224, height=224)


@pytest.fixture
def sample_image_large(rng):
    """生成大尺寸测试图像。"""
    return create_test_image(width=1920, height=1080)


@pytest.fixture
def sample_image_small(rng):
    """生成小尺寸测试图像。"""
    return create_test_image(width=56, height=56)


@pytest.fixture
def sample_image_path(tmp_path, sample_image):
    """创建临时图像文件路径。"""
    image_path = tmp_path / "test_image.png"
    sample_image.save(image_path)
    return str(image_path)


@pytest.fixture
def sample_conversation():
    """生成标准格式的对话数据。"""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "image", "image": "test_image.png"},
            ],
        }
    ]


@pytest.fixture
def sample_conversation_with_video():
    """生成包含视频的对话数据。"""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this video"},
                {"type": "video", "video": "test_video.mp4"},
            ],
        }
    ]


@pytest.fixture
def mock_processor():
    """Mock transformers processor。"""
    from unittest.mock import MagicMock

    processor = MagicMock()
    processor.apply_chat_template = MagicMock(
        return_value="<|im_start|>user\nHello<|im_end|>"
    )
    return processor

