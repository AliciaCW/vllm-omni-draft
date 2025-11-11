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

## 4. 用例
- 初始化：未传入 `stage_configs` 时应调用 `load_stage_configs_from_model` 并赋值；随后 `initialize_stages` 正确构建 `stage_list` 并完成 `set_engine`。
- 参数校验：`sampling_params_list` 与 `stage_list` 长度不一致时抛出 `ValueError`。
- 多阶段生成链路：
  - 第 0 阶段直接使用 `prompts`。
  - 后续阶段需调用 `process_engine_inputs(self.stage_list, prompts)` 的返回值作为输入。
  - 每阶段生成结果传给 `set_engine_outputs`。
  - 若阶段 `final_output=True`，应在返回列表中追加对应 `OmniRequestOutput`。
- 生成聚合：`_run_generation` 应将 `stage.engine.generate` 产生的迭代结果按顺序聚合为列表。

## 5. 目录与运行
- 测试文件：`src/CI_test/tests/test_omni_llm.py`
- 运行方式：在仓库根目录执行 `pytest -q src/CI_test/tests/test_omni_llm.py`

## 6. 验收标准
- 覆盖上述关键路径，避免真实重依赖。
- 本地与CI均可稳定、快速运行。


