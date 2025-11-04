"""基础测试示例，确保 CI 可以正常运行。"""

import pytest


def test_example():
    """示例测试，确保 pytest 可以正常运行。"""
    assert 1 + 1 == 2


@pytest.mark.slow
def test_slow_example():
    """慢测试示例，会被 CI 跳过。"""
    assert True

