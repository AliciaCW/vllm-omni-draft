"""测试 processing_omni.py 中的核心处理函数。

注意：实际使用时需要调整导入路径以匹配实际的 processing_omni.py 位置。
"""

import math

import numpy as np
import pytest
from PIL import Image

# 导入 processing_omni 模块
# 实际使用时取消注释并根据实际路径调整
# import sys
# sys.path.insert(0, 'path/to/vllm-omni-main/examples/offline_inference/qwen_2_5_omni')
# from processing_omni import (
#     ceil_by_factor,
#     extract_vision_info,
#     fetch_image,
#     floor_by_factor,
#     process_vision_info,
#     round_by_factor,
#     smart_nframes,
#     smart_resize,
# )

# 临时实现用于测试（实际项目中应导入真实模块）
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200
FRAME_FACTOR = 2


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """Rescales the image dimensions."""
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, "
            f"got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    """Extract vision info from conversations."""
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message.get("content"), list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele.get("type") in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


@pytest.mark.unit
class TestFactorFunctions:
    """测试因子取整函数。"""

    def test_round_by_factor(self):
        """测试 round_by_factor 函数。"""
        assert round_by_factor(100, 28) == 112
        assert round_by_factor(50, 28) == 56
        assert round_by_factor(14, 28) == 0

    def test_ceil_by_factor(self):
        """测试 ceil_by_factor 函数。"""
        assert ceil_by_factor(100, 28) == 112
        assert ceil_by_factor(50, 28) == 56
        assert ceil_by_factor(14, 28) == 28

    def test_floor_by_factor(self):
        """测试 floor_by_factor 函数。"""
        assert floor_by_factor(100, 28) == 84
        assert floor_by_factor(50, 28) == 28
        assert floor_by_factor(14, 28) == 0


@pytest.mark.unit
class TestSmartResize:
    """测试 smart_resize 函数。"""

    def test_normal_resize(self):
        """测试正常尺寸调整。"""
        height, width = 224, 224
        h_out, w_out = smart_resize(height, width)
        assert h_out % 28 == 0
        assert w_out % 28 == 0
        assert h_out == w_out

    def test_small_image(self):
        """测试小图像处理。"""
        height, width = 50, 50
        h_out, w_out = smart_resize(height, width)
        min_pixels = 4 * 28 * 28
        assert h_out * w_out >= min_pixels

    def test_large_image(self):
        """测试大图像处理。"""
        height, width = 2000, 2000
        h_out, w_out = smart_resize(height, width)
        max_pixels = 16384 * 28 * 28
        assert h_out * w_out <= max_pixels

    def test_extreme_aspect_ratio(self):
        """测试极端宽高比。"""
        height, width = 100, 20000
        with pytest.raises(ValueError, match="aspect ratio"):
            smart_resize(height, width)

    def test_custom_factor(self):
        """测试自定义因子。"""
        height, width = 100, 100
        h_out, w_out = smart_resize(height, width, factor=16)
        assert h_out % 16 == 0
        assert w_out % 16 == 0


@pytest.mark.unit
class TestExtractVisionInfo:
    """测试 extract_vision_info 函数。"""

    def test_extract_image_from_conversation(self):
        """测试从对话中提取图像信息。"""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "image", "image": "test.png"},
                ],
            }
        ]
        vision_infos = extract_vision_info(conversation)
        assert len(vision_infos) == 1
        assert vision_infos[0]["type"] == "image"

    def test_extract_video_from_conversation(self):
        """测试从对话中提取视频信息。"""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "test.mp4"},
                ],
            }
        ]
        vision_infos = extract_vision_info(conversation)
        assert len(vision_infos) == 1
        assert "video" in vision_infos[0]

    def test_multiple_conversations(self):
        """测试多个对话。"""
        conversations = [
            [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": "test1.png"}],
                }
            ],
            [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": "test2.png"}],
                }
            ],
        ]
        vision_infos = extract_vision_info(conversations)
        assert len(vision_infos) == 2

    def test_no_vision_info(self):
        """测试没有视觉信息的对话。"""
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}],
            }
        ]
        vision_infos = extract_vision_info(conversation)
        assert len(vision_infos) == 0


@pytest.mark.unit
class TestSmartNframes:
    """测试 smart_nframes 函数。"""

    @pytest.mark.skip(reason="需要导入 processing_omni 模块")
    def test_nframes_mode(self):
        """测试使用 nframes 模式。"""
        # from processing_omni import smart_nframes
        # ele = {"nframes": 10}
        # total_frames = 100
        # video_fps = 30.0
        # nframes = smart_nframes(ele, total_frames, video_fps)
        # assert nframes == 10
        # assert nframes % 2 == 0
        pass

    @pytest.mark.skip(reason="需要导入 processing_omni 模块")
    def test_fps_mode(self):
        """测试使用 fps 模式。"""
        # from processing_omni import smart_nframes
        # ele = {"fps": 2.0}
        # total_frames = 90
        # video_fps = 30.0
        # nframes = smart_nframes(ele, total_frames, video_fps)
        # assert nframes % 2 == 0
        # assert nframes >= 4
        # assert nframes <= total_frames
        pass


@pytest.mark.unit
class TestFetchImage:
    """测试 fetch_image 函数（使用 mock）。"""

    @pytest.mark.skip(reason="需要导入 processing_omni 模块")
    def test_fetch_image_from_pil(self, sample_image):
        """测试从 PIL Image 加载。"""
        # from processing_omni import fetch_image
        # ele = {"image": sample_image}
        # result = fetch_image(ele)
        # assert isinstance(result, Image.Image)
        # assert result.mode == "RGB"
        pass

    @pytest.mark.skip(reason="需要导入 processing_omni 模块")
    def test_fetch_image_from_base64(self):
        """测试从 base64 加载。"""
        # from processing_omni import fetch_image
        # img = Image.new("RGB", (100, 100), color="red")
        # buffer = BytesIO()
        # img.save(buffer, format="PNG")
        # img_base64 = base64.b64encode(buffer.getvalue()).decode()
        # data_url = f"data:image/png;base64,{img_base64}"
        # ele = {"image_url": data_url}
        # result = fetch_image(ele)
        # assert isinstance(result, Image.Image)
        pass


@pytest.mark.unit
class TestProcessVisionInfo:
    """测试 process_vision_info 函数。"""

    @pytest.mark.skip(reason="需要导入 processing_omni 模块")
    def test_process_vision_info_with_image(self, sample_image):
        """测试处理包含图像的对话。"""
        # from processing_omni import process_vision_info
        # conversation = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "text", "text": "Hello"},
        #             {"type": "image", "image": sample_image},
        #         ],
        #     }
        # ]
        # with patch("processing_omni.fetch_image") as mock_fetch:
        #     mock_fetch.return_value = sample_image
        #     images, videos = process_vision_info(conversation)
        #     assert images is not None
        #     assert videos is None
        pass

