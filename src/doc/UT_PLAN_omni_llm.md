# OmniLLM 单元测试设计（聚焦版）

本文档仅聚焦 `vllm-omni/vllm_omni/entrypoints/omni_llm.py` 中的 `OmniLLM` 类，目标是在不加载真实引擎/模型的前提下，验证初始化与生成流程的核心行为。

## 1. 测试目标
- 验证阶段配置加载与阶段实例化是否正确。
- 验证 `generate` 的参数校验、阶段输入处理与最终输出聚合逻辑。
- 验证 `_run_generation` 对阶段引擎产出的聚合行为。

## 2. 主要函数
- `__init__`（含 `initialize_stage_configs`、`initialize_stages` 的调用链）
- `initialize_stage_configs`
- `initialize_stages`
- `generate`
- `_run_generation`

## 3. 策略
- 使用 `pytest` 与 `monkeypatch`/`pytest-mock`。
- 通过 Fake/Mock 替代：
  - `load_stage_configs_from_model`：返回可控的伪 `stage_configs` 列表（含 `engine_args`、`final_output`、`final_output_type`）。
  - `OmniStage`：替换为可观察的轻量Fake，实现 `set_engine`、`set_engine_outputs`、`process_engine_inputs` 与可配置属性 `final_output`。
  - `OmniStageLLM`：替换为轻量Fake，提供 `generate`（返回可迭代）即可，避免真实引擎初始化。
- 辅助类：
  - `_FakeEngineArgs`：模拟 `engine_args`，同时支持字典和对象属性访问
  - `_FakeStageConfig`：模拟 `stage_config`，提供测试所需的配置结构

## 4. 测试替身类说明

### `_FakeEngineArgs`（引擎参数替身）
- **作用**：模拟 `engine_args` 对象，需要同时支持字典访问（`**kwargs`）和对象属性访问（`obj.attr`）
- **实现**：
  - 继承自 `dict`，支持字典操作
  - 自动设置必需属性：`model_stage`、`engine_output_type`（如果不存在则设为 `None`）
  - 将所有键值对同时设置为对象属性，支持 `obj.key` 访问方式
  - 这样可以在 `OmniStageLLM(model=model, **stage_config.engine_args)` 中作为 `**kwargs` 解包，同时 `OmniStage` 也可以使用对象属性访问

### `_FakeStageConfig`（阶段配置替身）
- **作用**：模拟 `stage_config` 对象，提供测试所需的配置结构
- **实现**：
  - `engine_args`：转换为 `_FakeEngineArgs` 对象，支持字典和对象两种访问方式
  - `final_output`：布尔值，控制是否作为最终输出
  - `final_output_type`：最终输出类型（如 `"text"`）
  - `stage_id`：阶段 ID（默认为 0）
  - `_config_dict`：存储原始配置字典，用于 `_FakeStage.process_engine_inputs` 返回预设值

### `_FakeStage`（替代 `OmniStage`）
- **作用**：替代真实的 `OmniStage`，避免加载真实引擎和模型权重
- **实现**：
  - `set_engine(engine)`：记录注入的引擎
  - `set_engine_outputs(outputs)`：记录生成结果，便于断言
  - `process_engine_inputs(stage_list, prompts)`：返回预设的处理结果，验证调用链
  - `final_output` / `final_output_type`：可配置属性，控制是否作为最终输出

### `_FakeEngine`（内部引擎替身）
- **作用**：`_FakeStageLLM` 内部使用，提供 `generate` 方法的迭代器接口
- **实现**：
  - `generate(prompts, sampling_params)`：返回预设的输出列表（可迭代）
  - `_last_prompts`：记录最近一次接收的 prompts，便于验证输入传递

### `_FakeStageLLM`（替代 `OmniStageLLM`）
- **作用**：替代真实的 `OmniStageLLM`，避免初始化真实的 vLLM 引擎（`LLMEngine.from_engine_args`）
- **实现**：
  - `__init__(**kwargs)`：接受参数但不初始化真实引擎，内部创建 `_FakeEngine`
  - `generate(prompts, sampling_params)`：委托给 `_FakeEngine.generate`，返回迭代器

## 5. 用例
- 初始化：未传入 `stage_configs` 时应调用 `load_stage_configs_from_model` 并赋值；随后 `initialize_stages` 正确构建 `stage_list` 并完成 `set_engine`。
- 参数校验：`sampling_params_list` 与 `stage_list` 长度不一致时抛出 `ValueError`。
- 多阶段生成链路：
  - 第 0 阶段直接使用 `prompts`。
  - 后续阶段需调用 `process_engine_inputs(self.stage_list, prompts)` 的返回值作为输入。
  - 每阶段生成结果传给 `set_engine_outputs`。
  - 若阶段 `final_output=True`，应在返回列表中追加对应 `OmniRequestOutput`。
- 生成聚合：`_run_generation` 应将 `stage.engine.generate` 产生的迭代结果按顺序聚合为列表。

## 6. 目录与运行
- 测试文件：`tests/examples/test_omni_llm.py`
- 运行方式：在仓库根目录执行 `pytest -q tests/examples/test_omni_llm.py`

## 7. 验收标准
- 覆盖上述关键路径，避免真实重依赖。
- 本地与CI均可稳定、快速运行。

## 8. 实现对比与测试用例详解

### 8.1 计划 vs 实现

✅ **一致**：所有计划中的 Mock 都已实现
- `load_stage_configs_from_model` → 通过 `monkeypatch.setattr` 替换为返回预设配置的函数
- `OmniStage` → `_FakeStage` 类，实现核心接口方法
- `OmniStageLLM` → `_FakeStageLLM` 类，避免真实引擎初始化

✅ **额外实现**：
- `_FakeEngineArgs`：计划中未明确提及但实际需要的辅助类，用于模拟 `engine_args`，同时支持字典和对象属性访问
- `_FakeStageConfig`：计划中未明确提及但实际需要的辅助类，用于模拟 `stage_config` 对象结构
- `_FakeEngine`：计划中未明确提及但实际需要的内部类，用于 `_FakeStageLLM` 内部实现，提供 `generate` 迭代器接口
- `_setup_engine_mocks(monkeypatch)`：辅助函数，统一设置引擎相关的 mocks，包括：
  - `LLMEngine.from_engine_args`：返回包含所有必要属性的 fake engine
  - `get_model_architecture` 和 `_get_model_architecture`：返回真实的 `FakeModelClass` 类（避免 `inspect.getsource` 问题）
  - `try_create_mm_pooling_model_cls`：直接返回传入的类
  - `_enable_processor_cache`：返回 `False`，避免设置 processor factories
  - `get_io_processor`：返回 `None`，避免 IO processor 初始化
- `mock_get_config` fixture（autouse=True）：自动 mock 模型加载相关函数，包括：
  - `vllm.transformers_utils.config.get_config`：返回 fake config
  - `transformers.utils.hub.cached_file` 和 `cached_files`：返回临时假配置文件，避免从 HuggingFace 下载

### 8.2 测试用例详细说明

#### 8.2.1 `test_initialize_stage_configs_called_when_none`

**测试目标**：验证当 `OmniLLM.__init__` 未传入 `stage_configs` 时，会自动调用 `load_stage_configs_from_model` 并正确初始化阶段列表。

**测试步骤**：
1. 删除模块缓存（`vllm_omni.entrypoints.utils`、`vllm_omni.entrypoints.omni_llm`、`vllm_omni.entrypoints.omni_stage`）以确保干净状态
2. 调用 `_setup_engine_mocks(monkeypatch)` 设置通用的引擎 mocks
3. Mock `load_stage_configs_from_model` 返回两个相同的配置（在定义处和导入处都 mock）
4. Mock `OmniStage` 和 `OmniStageLLM` 为 Fake 实现
5. 导入 `omni_llm_module` 并在模块内 patch `load_stage_configs_from_model`、`OmniStage` 和 `OmniStageLLM`（确保 mock 在模块导入后生效）
6. 创建 `OmniLLM(model="any")`，不传入 `stage_configs`
7. 验证 `llm.stage_configs` 是列表且长度为 2
8. 验证 `llm.stage_list` 长度为 2
9. 验证每个 stage 都是 `_FakeStage` 实例
10. 验证每个 stage 的 `engine` 都是 `_FakeStageLLM` 实例

**对应实现逻辑**（`omni_llm.py:30-36`）：
```python
if stage_configs is None:
    self.initialize_stage_configs(model)  # 调用 load_stage_configs_from_model
else:
    self.stage_configs = stage_configs
self.stage_list = []
self.initialize_stages(model)  # 创建 OmniStage 并设置 engine
```

**验证点**：
- ✅ 自动加载配置机制正确
- ✅ 阶段列表构建正确
- ✅ 引擎注入机制正确

---

#### 8.2.2 `test_generate_raises_on_length_mismatch`

**测试目标**：验证 `generate` 方法在 `sampling_params_list` 长度与 `stage_list` 长度不匹配时抛出 `ValueError`。

**测试步骤**：
1. 删除模块缓存以确保干净状态
2. 调用 `_setup_engine_mocks(monkeypatch)` 设置通用的引擎 mocks
3. Mock `load_stage_configs_from_model` 返回 1 个配置
4. Mock `OmniStage` 和 `OmniStageLLM` 为 Fake 实现
5. 创建包含 1 个 stage 的 `OmniLLM`
6. 调用 `generate` 时传入空的 `sampling_params_list=[]`
7. 验证抛出 `ValueError`

**对应实现逻辑**（`omni_llm.py:57-61`）：
```python
if len(sampling_params_list) != len(self.stage_list):
    raise ValueError(
        f"Expected {len(self.stage_list)} sampling params, "
        f"got {len(sampling_params_list)}"
    )
```

**验证点**：
- ✅ 参数校验逻辑正确
- ✅ 错误信息包含预期和实际长度

---

#### 8.2.3 `test_generate_pipeline_and_final_outputs`

**测试目标**：验证多阶段生成流程的完整链路，包括：
- 第 0 阶段直接使用原始 `prompts`
- 后续阶段通过 `process_engine_inputs` 处理输入
- 每阶段生成结果正确写入 `engine_outputs`
- `final_output=True` 的阶段正确聚合到最终输出

**测试步骤**：
1. 删除模块缓存以确保干净状态
2. 调用 `_setup_engine_mocks(monkeypatch)` 设置通用的引擎 mocks
3. 创建包含 2 个 stage 的配置，都设置 `final_output=True`，stage 1 的 `processed_input` 设置为 `["processed-for-stage-1"]`
4. Mock `load_stage_configs_from_model` 返回这两个配置
5. Mock `OmniStage` 和 `OmniStageLLM` 为 Fake 实现
6. 导入 `omni_llm_module` 并在模块内 patch 相关引用
7. 创建包含 2 个 stage 的 `OmniLLM`
8. 为每个 stage 注入不同的 Fake 引擎输出（`s0` 和 `s1`），直接替换 `llm.stage_list[i].engine`
9. 调用 `generate` 传入 1 个 prompt 和 2 个 sampling_params
10. 验证返回 2 个 `OmniRequestOutput`（因为两个 stage 都是 final_output）
11. 验证 stage 0 的 `_outputs` 包含 `[{"stage": 0, "text": "s0"}]`
12. 验证 stage 1 的 `_outputs` 包含 `[{"stage": 1, "text": "s1"}]`
13. 验证 stage 0 的 engine 接收了原始 `prompts`（通过 `s0._fake_engine._last_prompts`）
14. 验证 stage 1 的 engine 被调用（通过 `hasattr` 检查 `_last_prompts`）

**对应实现逻辑**（`omni_llm.py:62-79`）：
```python
for stage_id, stage in enumerate(self.stage_list):
    if stage_id > 0:
        engine_inputs = stage.process_engine_inputs(self.stage_list, prompts)  # 后续阶段处理输入
    else:
        engine_inputs = prompts  # 第 0 阶段直接使用原始 prompts
    engine_outputs = self._run_generation(stage, sampling_params_list[stage_id], engine_inputs)
    stage.set_engine_outputs(engine_outputs)  # 写入生成结果
    if hasattr(stage, "final_output") and stage.final_output:
        final_outputs.append(OmniRequestOutput(...))  # 聚合最终输出
```

**验证点**：
- ✅ 第 0 阶段使用原始 prompts
- ✅ 后续阶段调用 `process_engine_inputs`
- ✅ 生成结果正确写入 stage
- ✅ `final_output=True` 的阶段正确聚合
- ✅ `_run_generation` 正确收集迭代器结果

---

#### 8.2.4 `test_generate_no_final_output_returns_empty`

**测试目标**：验证当所有 stage 的 `final_output=False` 时，`generate` 返回空列表。

**测试步骤**：
1. 删除模块缓存以确保干净状态
2. 调用 `_setup_engine_mocks(monkeypatch)` 设置通用的引擎 mocks
3. 创建包含 2 个 stage 的配置，都设置 `final_output=False`
4. Mock `load_stage_configs_from_model` 返回这两个配置
5. Mock `OmniStage` 和 `OmniStageLLM` 为 Fake 实现
6. 导入 `omni_llm_module` 并在模块内 patch 相关引用
7. 创建包含 2 个 stage 的 `OmniLLM`
8. 调用 `generate` 传入 prompts 和 2 个 sampling_params
9. 验证返回空列表 `[]`

**对应实现逻辑**（`omni_llm.py:71-78`）：
```python
if hasattr(stage, "final_output") and stage.final_output:
    final_outputs.append(OmniRequestOutput(...))
# 如果所有 stage 的 final_output 都是 False，final_outputs 保持为空列表
```

**验证点**：
- ✅ `final_output=False` 的阶段不加入最终输出
- ✅ 所有阶段都不是 final_output 时返回空列表

---

#### 8.2.5 `test_generate_sampling_params_none_raises`

**测试目标**：验证当 `sampling_params_list=None` 时，`generate` 会抛出异常。

**测试步骤**：
1. 删除模块缓存以确保干净状态
2. 调用 `_setup_engine_mocks(monkeypatch)` 设置通用的引擎 mocks
3. Mock `load_stage_configs_from_model` 返回 1 个配置
4. Mock `OmniStage` 和 `OmniStageLLM` 为 Fake 实现
5. 创建包含 1 个 stage 的 `OmniLLM`
6. 调用 `generate` 时传入 `sampling_params_list=None`
7. 验证抛出异常（`TypeError` 或 `ValueError`，使用 `pytest.raises(Exception)` 捕获）

**对应实现逻辑**（`omni_llm.py:57`）：
```python
if len(sampling_params_list) != len(self.stage_list):  # 如果 sampling_params_list 是 None，这里会抛出 TypeError
    raise ValueError(...)
```

**验证点**：
- ✅ `None` 参数被正确检测并抛出异常
- ⚠️ **注意**：实际代码在 `len(None)` 时会抛出 `TypeError`，测试使用 `pytest.raises(Exception)` 捕获，这是合理的

---

### 8.3 实现正确性检查

#### 8.3.1 Fake 配置结构验证

**`fake_stage_config` 结构**：
```python
{
    "engine_args": {},                        # ✅ 正确：对应 stage_config.engine_args（空字典，因为 model 单独传递）
    "final_output": True,                     # ✅ 正确：对应 stage_config.final_output
    "final_output_type": "text",             # ✅ 正确：对应 stage_config.final_output_type
    "processed_input": ["processed-by-stage"], # ✅ 正确：用于 _FakeStage.process_engine_inputs 返回
}
```

**验证**：
- ✅ `engine_args` 会被解包传给 `OmniStageLLM(model=model, **stage_config.engine_args)`（注意：`model` 参数单独传递，不在 `engine_args` 中）
- ✅ `final_output` 和 `final_output_type` 会被 `_FakeStage` 读取并设置
- ✅ `processed_input` 是测试辅助字段，用于验证 `process_engine_inputs` 调用链

#### 8.3.2 函数名和参数验证

| 测试中的调用                                       | 实际实现                                                | 状态                                   |
| -------------------------------------------------- | ------------------------------------------------------- | -------------------------------------- |
| `OmniStage(cfg)`                                   | `OmniStage(stage_config)`                               | ✅ 正确（通过 monkeypatch 替换）        |
| `OmniStageLLM(**kw)`                               | `OmniStageLLM(model=model, **stage_config.engine_args)` | ✅ 正确（kw 包含 model 和 engine_args） |
| `stage.set_engine(engine)`                         | `stage.set_engine(engine)`                              | ✅ 正确                                 |
| `stage.set_engine_outputs(outputs)`                | `stage.set_engine_outputs(engine_outputs)`              | ✅ 正确                                 |
| `stage.process_engine_inputs(stage_list, prompts)` | `stage.process_engine_inputs(stage_list, prompt)`       | ✅ 正确（参数名略有不同但不影响）       |
| `stage.engine.generate(prompts, sampling_params)`  | `stage.engine.generate(prompts, sampling_params)`       | ✅ 正确                                 |

#### 8.3.3 Mock 设置策略

**关键实现细节**：

1. **模块缓存清理**：
   - 在设置 mocks 之前，先删除 `sys.modules` 中的相关模块缓存
   - 确保模块重新导入时能使用新的 mocks
   - 涉及的模块：`vllm_omni.entrypoints.utils`、`vllm_omni.entrypoints.omni_llm`、`vllm_omni.entrypoints.omni_stage`

2. **Mock 设置顺序**：
   - 先调用 `_setup_engine_mocks(monkeypatch)` 设置通用的引擎相关 mocks
   - 然后 mock `load_stage_configs_from_model`、`OmniStage`、`OmniStageLLM`
   - 导入 `omni_llm_module` 后，再次在模块内 patch 这些引用（确保模块内导入的引用也被替换）

3. **为什么需要双重 Mock**：
   - 在模块定义处 mock（如 `vllm_omni.entrypoints.omni_stage.OmniStage`）：影响新导入的模块
   - 在模块导入后 mock（如 `omni_llm_module.OmniStage`）：影响已导入模块内的引用
   - 这样可以确保无论代码从哪个位置引用这些类，都能使用 mock 版本

#### 8.3.4 潜在问题检查

1. **`sampling_params_list=None` 的处理**：
   - 实际代码：`len(sampling_params_list)` 在 `None` 时会抛出 `TypeError`
   - 测试：使用 `pytest.raises(Exception)` 捕获，可以接受
   - ⚠️ **建议**：实际代码应该先检查 `if sampling_params_list is None`，但这是实现问题，不是测试问题

2. **`_FakeStage` 缺少的属性**：
   - 实际 `OmniStage` 有 `stage_id`, `engine_args`, `model_stage` 等属性
   - `_FakeStage` 只实现了测试需要的接口，这是合理的（轻量级 Mock）

3. **`process_engine_inputs` 的返回值**：
   - 实际实现返回 `List[Union[OmniTokensPrompt, TextPrompt]]`
   - `_FakeStage` 返回预设的 `_processed_input`（列表），类型兼容

4. **模型加载相关的 Mock**：
   - `mock_get_config` fixture（autouse=True）自动 mock 所有测试中的模型加载函数
   - `_setup_engine_mocks` 统一设置引擎初始化相关的 mocks
   - 这些 mocks 避免了真实的模型权重加载和引擎初始化

### 8.4 总结

✅ **所有测试用例实现正确**：
- Mock 替换机制正确，包括模块定义处和模块导入后的双重 mock
- 函数调用和参数传递正确
- Fake 配置结构符合实际需求
- 测试覆盖了关键路径和边界情况
- 通过 `_setup_engine_mocks` 和 `mock_get_config` fixture 统一管理 mocks，避免重复代码

✅ **无臆想部分**：
- 所有 Mock 都基于实际代码接口
- 所有断言都验证实际行为
- 没有测试不存在的功能

✅ **Mock 策略优化**：
- 使用辅助函数 `_setup_engine_mocks` 统一设置引擎相关 mocks，提高代码复用性
- 使用 `autouse=True` 的 fixture `mock_get_config` 自动处理模型加载相关的 mocks
- 在模块导入后再次 patch 模块内引用，确保 mock 完全生效

⚠️ **代码改进建议**（非测试问题）：
- `generate` 方法应该先检查 `sampling_params_list is None`，再检查长度
