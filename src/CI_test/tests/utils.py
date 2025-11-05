"""测试工具函数，用于生成测试数据。"""

import numpy as np
from PIL import Image


def random_image(rng: np.random.RandomState, min_wh: int, max_wh: int) -> Image.Image:
    """生成随机尺寸的测试图像。

    Args:
        rng: 随机数生成器
        min_wh: 最小宽高
        max_wh: 最大宽高

    Returns:
        随机生成的 PIL Image
    """
    w, h = rng.randint(min_wh, max_wh, size=(2,))
    arr = rng.randint(0, 255, size=(h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def create_test_image(width: int, height: int) -> Image.Image:
    """创建指定尺寸的测试图像。

    Args:
        width: 图像宽度
        height: 图像高度

    Returns:
        指定尺寸的 PIL Image
    """
    arr = np.random.randint(0, 255, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def create_test_video_array(
    num_frames: int, height: int, width: int
) -> np.ndarray:
    """创建测试视频数组。

    Args:
        num_frames: 帧数
        height: 帧高度
        width: 帧宽度

    Returns:
        视频数组，形状为 (num_frames, height, width, 3)
    """
    return np.random.randint(0, 255, size=(num_frames, height, width, 3), dtype=np.uint8)

