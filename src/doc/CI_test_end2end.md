### 目标

有参考 vllm/tests/models/multimodal/generation/test_qwen2_vl.py文件。

为 `vllm-omni/examples/offline_inference/qwen_2_5_omni/end2end.py` 设计一组可在 CI 执行的测试计划，在不加载真实权重的前提下验证：参数解析、环境变量/随机性设置、调用编排、输出落盘分支的正确性与稳定性。

### 参考的 vLLM 测试风格要点

- 基于 pytest，充分利用夹具与标记：
  - 全局 `conftest.py` 自动清理/设置环境变量（如 `VLLM_USE_V1`）、分布式上下文、日志；
  - 通过 `monkeypatch` 注入/替换重资源逻辑或外部依赖；
  - 对外部服务/权重下载使用“假路径/假实现”以保证 CI 可跑；
  - 使用 `@pytest.mark.skip`/自定义标记控制在资源不足场景下跳过；
  - 结果校验强调“确定性”（固定 `seed`、关闭随机采样）。

### 测试思路（跳过权重加载）

- 不直接运行真实 `OmniLLM` 推理；通过 `monkeypatch` 将 `OmniLLM` 类替换为轻量假的实现，拦截 `generate`，返回模拟输出：
  - `final_output_type` 分支覆盖：`text` 与 `audio`；
  - 返回值中包含最小必要字段：`request_output` 列表、其中每个 `output` 含 `request_id`、`outputs[0].text` 或 `multimodal_output["audio"]`（`torch.Tensor`）等；
  - 固定随机种子情况下，伪实现返回确定性内容（依赖入参，或固定模板）。
- 拦截文件写入：`soundfile.write` 替换为“记录调用但不落盘”的假函数，校验被调用次数、采样率、路径拼接；
- 通过 `capsys` 驱动脚本入口 `main()`/`argparse`，必要时再用子进程。校验标准输出格式及关键字段出现（`Request ID`, `Saved audio to ...`）。
- 对环境变量副作用：
  - 脚本顶部设置 `VLLM_USE_V1=1`，测试内使用 `monkeypatch` 与 vLLM 的自动清理夹具协同，确保用例间互不污染；
  - 校验 `SEED` 相关设置不抛异常（不做数值等值断言，避免与平台差异耦合）。

### 用例清单与优先级

- 必需
  - 参数解析最小集：
    - 仅提供 `--model` 与 `--prompts`（文本），校验 `argparse` 正常通过；
  - 纯文本路径：
    - `prompt_type=text`，假 `OmniLLM.generate` 返回 `final_output_type=text`；
    - 断言 stdout 打印包含 `Request ID:` 与生成文本，`sf.write` 未被调用；
  - 音频输出路径：
    - 传入 `--do-wave` 与 `--output-wav <dir>`；假 `generate` 返回 `final_output_type=audio`，`multimodal_output["audio"]` 为 `(N, )` 的 `torch.float32/float16`；
    - 断言 `sf.write` 被调用 N 次、采样率为 `24000`，输出路径拼接正确（包含 `output_<request_id>.wav`）；
  - 多阶段参数打包：
    - 校验脚本对 `SamplingParams` 列表创建成功，并按顺序传入假 `OmniLLM.generate`（通过假实现记录调用参数长度/顺序并断言）；
  - 环境变量/种子设置：
    - 断言运行过程中不抛异常；`VLLM_USE_V1` 在测试退出后被还原（依赖全局夹具自动清理即可）。

- 可选
  - `--tokenize` 分支：
    - 打开后不影响 `generate` 的调用与输出（仅验证未抛异常、调用路径一致）；
  - `--thinker-only`/`--text-only`：
    - 覆盖采样参数列表长度与分支控制（假实现校验收到的 `sampling_params_list` 大小变化）；
  - `--voice-type` 与 `--code2wav-*` 参数传递：
    - 校验解析与传递到假 `OmniLLM` 的调用记录中（不做语义校验）；
  - 批量多 `--prompts`：
    - 返回多个 `request_output`，校验打印/写文件次数与顺序；
  - `--output-wav` 为已存在文件路径 vs 目录：
    - 校验脚本中 `os.makedirs(..., exist_ok=True)` 的健壮性（当作目录使用）；
  - `--prompt_type` 覆盖更多占位类型（如 `audio`, `image`）：
    - 仅验证控制流不崩溃；假实现忽略具体模态内容。

### 实现要点

- 入口调用：
  - 首选直接导入模块后调用 `main()`，利用 `monkeypatch.setenv` 与 `monkeypatch.setattr` 定向替换；
  - 如需验证 `argparse` 行为，可用 `pytest` 的 `monkeypatch.setenv("PYTHONPATH", ...)`/`capsys` + 构造 `sys.argv`。
- 替换点：
  - `vllm_omni.entrypoints.omni_llm.OmniLLM` → 伪类：记录 `__init__(model=...)` 与 `generate(prompts, sampling_params_list)` 调用；
  - `soundfile.write` → 伪函数：收集 `(path, array.shape, samplerate)`；
  - 如需避免 `torch` 设备相关初始化，可在伪实现中全部用 CPU 张量。
- 断言范围：
  - 仅断言“是否被调用/参数个数/关键字段存在/路径格式正确”；
  - 避免对具体 token/text 的强绑定断言，保证 CI 稳定。

### 目录与命名建议（后续实现时）

- 放置位置：`vllm-omni/examples/offline_inference/qwen_2_5_omni/tests/test_end2end.py` 或统一集中到本仓库已有 CI 测试目录（视现有测试布局而定）。
- 标记：可加 `@pytest.mark.optional` 或自定义标记，默认在无权重/无 GPU 的 CI 上可运行。

### 验收标准

- 在无网络/无 GPU 的 CI 环境可稳定通过；
- 不依赖真实权重；
- 能覆盖主要控制流（文本/音频、参数/环境变量设置、批量多请求）。

### 夹具与参数化（与 vllm/tests 一致）

- 环境变量 fixture（function 级，autouse）：
  - 用 `monkeypatch.setenv('VLLM_USE_V1', '1')` 显式设定；
  - 测试退出后由 pytest 自动恢复，无交叉污染。
- 参数化：
  - 分支项统一用 `@pytest.mark.parametrize` 组合：
    - 输出类型：`['text', 'audio']`
    - prompts 数量：`[1, 2]`
    - 旗标：`thinker_only=[False, True]`、`text_only=[False, True]`、`tokenize=[False, True]`
  - 文件/网络重依赖的用例加 `@pytest.mark.optional`，默认跳过。

### 入口与断言

- 入口调用：导入模块，注入 `sys.argv` 后直接调用 `main()`；用 `capsys` 捕获 stdout。
- 断言：
  - 文本分支：stdout 含 `Request ID:` 与非空文本；`sf.write` 未被调用。
  - 音频分支：`sf.write` 被调用 N 次，`samplerate=24000`，路径包含 `output_<request_id>.wav`。
- 采样参数：fake 记录 `sampling_params_list` 的长度与顺序并断言。
  - 不对具体文本内容/数值做强等断言。

### fake 输出结构示例（供实现参考）

```python
class _FakeStageOutputs:
    def __init__(self, final_output_type: str, num_requests: int):
        self.final_output_type = final_output_type  # 'text' | 'audio'
        self.request_output = []
        for i in range(num_requests):
            request_id = f"req_{i}"
            if final_output_type == 'text':
                self.request_output.append(type('R', (), {
                    'request_id': request_id,
                    'outputs': [type('O', (), {'text': f'text_{i}'})],
                }))
            else:
                import torch
                self.request_output.append(type('R', (), {
                    'request_id': request_id,
                    'multimodal_output': {
                        'audio': torch.zeros(24000, dtype=torch.float32)
                    },
                }))

class _FakeOmniLLM:
    def __init__(self, model: str):
        self.model = model
        self.calls = []

    def generate(self, prompts, sampling_params_list):
        self.calls.append((prompts, sampling_params_list))
        # 返回两个阶段，分别覆盖 text 与 audio 分支
        return [
            _FakeStageOutputs('text', num_requests=len(prompts)),
            _FakeStageOutputs('audio', num_requests=len(prompts)),
        ]
```

实现测试时，用 `monkeypatch.setattr('vllm_omni.entrypoints.omni_llm.OmniLLM', _FakeOmniLLM)` 与 `monkeypatch.setattr('soundfile.write', fake_write)` 进行替换。


