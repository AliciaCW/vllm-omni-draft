# Qwen2.5-Omni CI 测试计划

## 概述

本文档描述为 `processing_omni.py` 文件及 `qwen_2_5_omni/` 文件夹设计的 CI 集成测试方案。测试设计遵循精简原则，仅覆盖核心功能，同时具备良好的可扩展性，便于后续添加新模型时复用。

## 测试文件结构

### 1. `tests/examples/offline_inference/qwen_2_5_omni/test_processing_omni.py` (单元测试)

**功能**：测试 `processing_omni.py` 中的核心处理函数

**覆盖范围**：
- **图像处理函数**
  - `smart_resize()`: 测试图像尺寸调整逻辑，包括边界条件（min_pixels、max_pixels、aspect ratio）
  - `fetch_image()`: 测试图像加载（支持本地路径、PIL Image、base64 格式）
  - `round_by_factor()`, `ceil_by_factor()`, `floor_by_factor()`: 测试因子取整工具函数

- **视频处理函数**
  - `smart_nframes()`: 测试视频帧数计算逻辑（fps 和 nframes 两种模式）
  - `fetch_video()`: 测试视频加载（torchvision 和 decord 两种后端）
  - 视频尺寸调整逻辑

- **数据处理函数**
  - `extract_vision_info()`: 测试从对话中提取视觉信息
  - `process_vision_info()`: 测试完整的视觉信息处理流程

**测试特点**：
- 使用 mock 对象避免实际下载网络资源
- 使用临时文件测试本地文件处理
- 涵盖正常流程和异常情况（如无效输入、格式错误）

---

### 2. `tests/examples/offline_inference/qwen_2_5_omni/test_qwen2_5_omni_integration.py` (集成测试)

**功能**：测试 Qwen2.5-Omni 端到端推理流程

**覆盖范围**：
- **Prompt 构建**
  - `make_text_prompt()`: 测试文本 prompt 构建
  - `make_omni_prompt()`: 测试多模态 prompt 构建（文本、图像、视频）

- **模型加载和初始化**
  - `OmniLLM` 初始化（使用小模型或 mock）
  - 模型配置验证

- **推理流程**
  - 文本生成流程（最小化测试，不依赖完整模型）
  - 多模态输入处理流程

**测试特点**：
- 使用轻量级 mock 或最小配置避免 GPU 依赖
- 重点测试数据流和接口调用，而非完整模型推理
- 可配置跳过需要 GPU 的测试（通过 pytest marker）

---

### 3. `tests/examples/offline_inference/qwen_2_5_omni/conftest.py` (测试配置)

**功能**：提供共享的 pytest fixtures 和测试工具

**包含内容**：
- **测试数据 fixtures**
  - `sample_image()`: 生成测试用 PIL Image
  - `sample_video_path()`: 创建临时测试视频文件
  - `sample_audio_path()`: 创建临时测试音频文件
  - `sample_conversation()`: 生成标准格式的对话数据

- **Mock fixtures**
  - `mock_processor()`: Mock transformers processor
  - `mock_omni_llm()`: Mock OmniLLM 实例

---

### 4. `tests/examples/offline_inference/qwen_2_5_omni/utils.py` (测试工具)

**功能**：提供测试数据生成和工具函数

**包含内容**：
- **数据生成函数**
  - `random_image()`: 生成随机尺寸的测试图像
  - `random_video()`: 生成随机帧数的测试视频数组
  - `random_audio()`: 生成随机长度的测试音频
  - `create_test_video_file()`: 从图像创建测试视频文件

- **测试辅助函数**
  - 图像/视频格式转换工具
  - 文件清理工具

---

### 5. `tests/examples/offline_inference/qwen_2_5_omni/assets/` (测试资源)

**功能**：存放静态测试资源文件

**包含内容**：
- `test_image.png`: 标准测试图像（用于测试图像加载）
- `test_video.mp4`: 短测试视频（用于测试视频加载）
- `test_audio.wav`: 测试音频文件（可选）

**用途**：
- 避免动态生成，提高测试稳定性
- 可用于测试网络下载场景（mock HTTP 响应）

---

## 测试文件总览

| 文件路径 | 文件数量 | 主要功能 | 测试类型 |
|---------|---------|---------|---------|
| `test_processing_omni.py` | 1 | 单元测试：图像/视频处理核心函数 | Unit |
| `test_qwen2_5_omni_integration.py` | 1 | 集成测试：端到端推理流程 | Integration |
| `conftest.py` | 1 | 测试配置：共享 fixtures 和工具 | Test Config |
| `utils.py` | 1 | 测试工具：生成测试数据的辅助函数 | Test Utils |
| `assets/` | 1 | 测试资源：静态图片/视频文件 | Test Assets |
| **总计** | **5** | - | - |

---

## 测试覆盖的关键功能点

### 核心功能（必须测试）
1. ✅ 图像尺寸调整和加载
2. ✅ 视频帧提取和尺寸调整
3. ✅ 多模态数据提取和处理
4. ✅ Prompt 构建流程

### 边界情况（必须测试）
1. ✅ 极端宽高比图像处理
2. ✅ 最小/最大像素限制
3. ✅ 视频帧数边界条件
4. ✅ 无效输入处理

### 可选功能（后续扩展）
- [ ] 音频处理测试
- [ ] 网络资源下载测试
- [ ] 完整模型推理测试（需要 GPU）

---

## 设计原则

### 1. 精简原则
- 只测试核心功能，避免过度测试
- 使用 mock 避免外部依赖（网络、GPU、大模型）
- 测试快速执行，适合 CI 环境

### 2. 可扩展性
- **模块化设计**：每个测试文件独立，便于单独运行
- **参数化测试**：使用 `@pytest.mark.parametrize` 支持多场景测试
- **清晰的结构**：测试目录结构可复制到其他模型文件夹
- **配置驱动**：通过 conftest.py 集中管理测试配置

### 3. 模板化
- 测试结构可作为其他模型的模板
- 只需修改模型特定的部分（如 prompt 格式、模型初始化）
- 保持测试命名和结构的一致性

---

## 后续扩展建议

当添加新模型（如 `model_x/`）时，可以：

1. **复制测试结构**：
   ```
   tests/examples/offline_inference/model_x/
   ├── test_processing_omni.py  # 如果复用 processing_omni.py
   ├── test_model_x_integration.py  # 模型特定的集成测试
   └── conftest.py  # 模型特定的 fixtures
   ```

2. **复用通用测试**：
   - `processing_omni.py` 相关的测试可以直接复用
   - 只需要新增模型特定的集成测试

3. **扩展测试场景**：
   - 根据新模型特性添加新的测试用例
   - 保持测试文件结构一致

---

## 测试运行方式

### 本地运行
```bash
# 运行所有测试
pytest tests/examples/offline_inference/qwen_2_5_omni/

# 运行单元测试
pytest tests/examples/offline_inference/qwen_2_5_omni/test_processing_omni.py -m unit

# 运行集成测试（跳过需要 GPU 的测试）
pytest tests/examples/offline_inference/qwen_2_5_omni/test_qwen2_5_omni_integration.py -m integration -m "not slow"

# 运行特定测试文件
pytest tests/examples/offline_inference/qwen_2_5_omni/test_processing_omni.py::test_smart_resize -v
```

### CI 运行
- 在 GitHub Actions 中配置 pytest 命令
- 使用 `pytest --cov` 生成覆盖率报告
- 标记需要 GPU 的测试为 `@pytest.mark.slow`，在 CI 中可选跳过

---

## 依赖要求

### 测试依赖（最小集）
- `pytest` >= 7.0
- `pytest-cov` (覆盖率报告)
- `pytest-mock` (Mock 支持)
- `Pillow` (图像处理测试)
- `torch` (视频处理测试，可选 CPU 版本)
- `torchvision` (视频处理测试)

### 可选依赖
- `decord` (视频后端测试，可选)
- `transformers` (集成测试，用于 mock)

---

## 注意事项

1. **避免真实模型加载**：在 CI 中使用 mock 或最小配置，避免下载大模型
2. **资源清理**：确保临时文件在测试后正确清理
3. **跨平台兼容**：注意 Windows/Linux 路径差异
4. **测试隔离**：每个测试应独立，不依赖执行顺序
5. **性能考虑**：测试应在合理时间内完成（建议 < 5 分钟）

---

## 实施优先级

### Phase 1（必须实现）
- [x] `test_processing_omni.py` - 核心处理函数测试
- [x] `conftest.py` - 基础 fixtures
- [x] `utils.py` - 测试工具函数

### Phase 2（重要）
- [ ] `test_qwen2_5_omni_integration.py` - 集成测试（最小化版本）
- [ ] `assets/` - 测试资源文件（至少 1 个图片用于基础测试）

### Phase 3（可选）
- [ ] 添加更多边界情况测试
- [ ] 性能基准测试
- [ ] 覆盖率报告优化
- [ ] `ci_envs.py` - CI 环境变量配置（用于控制测试行为）


