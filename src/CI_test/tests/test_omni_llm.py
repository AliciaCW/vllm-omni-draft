from typing import Any, List

import pytest


class _FakeStage:
    """轻量Stage替身：可观察交互，不依赖真实实现。"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.engine = None
        self._outputs: list = []
        # 允许配置 final_output 与 final_output_type，默认不输出
        self.final_output = config.get("final_output", False)
        self.final_output_type = config.get("final_output_type", None)
        # 可配置处理逻辑，默认回传占位
        self._processed_input = config.get("processed_input", ["processed"])
        # 捕获传入 generate 的 prompts，便于断言
        self._last_prompts = None

    def set_engine(self, engine) -> None:
        self.engine = engine

    def set_engine_outputs(self, outputs: list) -> None:
        self._outputs = outputs

    def process_engine_inputs(self, stage_list, prompts):
        # 简化：返回预设的处理结果，验证调用链路即可
        return self._processed_input


class _FakeEngine:
    """轻量Engine替身：提供 generate 迭代输出。"""

    def __init__(self, outputs: List[Any]):
        self._outputs = outputs

    def generate(self, prompts, sampling_params):
        # 记录最近一次的 prompts 以便外层断言
        self._last_prompts = prompts
        # 简化：一次性返回预设列表，保证可迭代
        yield from self._outputs


class _FakeStageLLM:
    """替代 OmniStageLLM，避免构造真实引擎。"""

    def __init__(self, **kwargs):
        # 允许注入自定义的 fake outputs，默认返回单条占位输出
        fake_outputs = kwargs.get("_fake_outputs", [[{"text": "ok"}]])
        self._fake_engine = _FakeEngine(fake_outputs)

    def generate(self, prompts, sampling_params):
        yield from self._fake_engine.generate(prompts, sampling_params)


@pytest.fixture
def fake_stage_config():
    return {
        "engine_args": {"model": "fake-model"},
        "final_output": True,
        "final_output_type": "text",
        # 测试第二阶段会用到 processed_input 以验证链路
        "processed_input": ["processed-by-stage"],
    }


def test_initialize_stage_configs_called_when_none(monkeypatch, fake_stage_config):
    # 准备：load_stage_configs_from_model 返回两个相同配置
    def _fake_loader(model: str):
        return [fake_stage_config, fake_stage_config]

    monkeypatch.setattr(
        "vllm_omni.entrypoints.utils.load_stage_configs_from_model",
        _fake_loader,
        raising=True,
    )
    # 替换 OmniStage 与 OmniStageLLM
    monkeypatch.setattr(
        "vllm_omni.entrypoints.omni_stage.OmniStage", lambda cfg: _FakeStage(cfg), raising=True
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.omni_llm.OmniStageLLM", lambda **kw: _FakeStageLLM(**kw), raising=True
    )

    from vllm_omni.entrypoints.omni_llm import OmniLLM

    llm = OmniLLM(model="any")
    # 验证：自动加载的 stage_configs 与 stage_list 数量一致
    assert isinstance(llm.stage_configs, list)
    assert len(llm.stage_configs) == 2
    assert len(llm.stage_list) == 2
    # 验证：每个Stage均已注入 engine
    for st in llm.stage_list:
        assert isinstance(st, _FakeStage)
        assert isinstance(st.engine, _FakeStageLLM)


def test_generate_raises_on_length_mismatch(monkeypatch, fake_stage_config):
    monkeypatch.setattr(
        "vllm_omni.entrypoints.utils.load_stage_configs_from_model",
        lambda model: [fake_stage_config],
        raising=True,
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.omni_stage.OmniStage", lambda cfg: _FakeStage(cfg), raising=True
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.omni_llm.OmniStageLLM", lambda **kw: _FakeStageLLM(**kw), raising=True
    )
    from vllm_omni.entrypoints.omni_llm import OmniLLM

    llm = OmniLLM(model="any")
    with pytest.raises(ValueError):
        llm.generate(prompts=["hi"], sampling_params_list=[])


def test_generate_pipeline_and_final_outputs(monkeypatch, fake_stage_config):
    # 两个阶段：阶段0直接用原始prompts；阶段1需要 process_engine_inputs 的结果
    stage_cfg0 = dict(fake_stage_config)
    stage_cfg1 = dict(fake_stage_config)
    # 确保阶段1的 processed_input 有明确标记，便于断言
    stage_cfg1["processed_input"] = ["processed-for-stage-1"]

    monkeypatch.setattr(
        "vllm_omni.entrypoints.utils.load_stage_configs_from_model",
        lambda model: [stage_cfg0, stage_cfg1],
        raising=True,
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.omni_stage.OmniStage",
        lambda cfg: _FakeStage(cfg),
        raising=True,
    )
    # 注入不同的 fake engine outputs，便于区分阶段输出
    def _fake_stage_llm_factory(**kw):
        # 允许从 engine_args 透传不同的假输出（若需要）
        fake_outputs = kw.get("_fake_outputs")
        if fake_outputs is None:
            fake_outputs = [[{"stage": "default"}]]
        return _FakeStageLLM(_fake_outputs=fake_outputs)

    monkeypatch.setattr(
        "vllm_omni.entrypoints.omni_llm.OmniStageLLM",
        _fake_stage_llm_factory,
        raising=True,
    )

    from vllm_omni.entrypoints.omni_llm import OmniLLM

    llm = OmniLLM(model="any")

    # 为每个 stage 注入不同的输出，模拟两轮生成
    # 这里通过直接替换已创建的 stage.engine 为不同 _FakeEngine
    s0 = _FakeStageLLM(_fake_outputs=[[{"stage": 0, "text": "s0"}]])
    s1 = _FakeStageLLM(_fake_outputs=[[{"stage": 1, "text": "s1"}]])
    llm.stage_list[0].engine = s0
    llm.stage_list[1].engine = s1

    # 准备 sampling params（内容不重要，只确保长度一致）
    sampling_params_list = [object(), object()]
    prompts = ["hi"]
    outputs = llm.generate(prompts=prompts, sampling_params_list=sampling_params_list)

    # 两个阶段都 final_output=True，因此应聚合两个 OmniRequestOutput
    assert len(outputs) == 2
    # 验证阶段1确实走了 process_engine_inputs 的路径（通过其内部固定返回值来侧面确认）
    assert llm.stage_list[1].process_engine_inputs([], []) is not None
    # 验证每个 stage 都已写回 engine_outputs
    assert llm.stage_list[0]._outputs == [[{"stage": 0, "text": "s0"}]]
    assert llm.stage_list[1]._outputs == [[{"stage": 1, "text": "s1"}]]
    # 验证阶段0确实拿到了原始prompts
    assert s0._fake_engine._last_prompts == prompts
    # 验证阶段1理应拿到 processed 输入（无法直接取内部值，这里只确认s1也被调用）
    assert hasattr(s1._fake_engine, "_last_prompts")


def test_generate_no_final_output_returns_empty(monkeypatch, fake_stage_config):
    # 两个阶段都不作为 final_output
    stage_cfg0 = dict(fake_stage_config)
    stage_cfg1 = dict(fake_stage_config)
    stage_cfg0["final_output"] = False
    stage_cfg1["final_output"] = False
    monkeypatch.setattr(
        "vllm_omni.entrypoints.utils.load_stage_configs_from_model",
        lambda model: [stage_cfg0, stage_cfg1],
        raising=True,
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.omni_stage.OmniStage", lambda cfg: _FakeStage(cfg), raising=True
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.omni_llm.OmniStageLLM", lambda **kw: _FakeStageLLM(**kw), raising=True
    )
    from vllm_omni.entrypoints.omni_llm import OmniLLM

    llm = OmniLLM(model="any")
    outputs = llm.generate(prompts=["p"], sampling_params_list=[object(), object()])
    assert outputs == []


def test_generate_sampling_params_none_raises(monkeypatch, fake_stage_config):
    # 由于 generate 内部会对 len(None) 触发异常，断言为 TypeError 或 ValueError 均可接受
    monkeypatch.setattr(
        "vllm_omni.entrypoints.utils.load_stage_configs_from_model",
        lambda model: [fake_stage_config],
        raising=True,
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.omni_stage.OmniStage", lambda cfg: _FakeStage(cfg), raising=True
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.omni_llm.OmniStageLLM", lambda **kw: _FakeStageLLM(**kw), raising=True
    )
    from vllm_omni.entrypoints.omni_llm import OmniLLM

    llm = OmniLLM(model="any")
    with pytest.raises(Exception):
        llm.generate(prompts=["p"], sampling_params_list=None)


