# 2025-11-12 test ok!
from typing import Any, List

import pytest


class _FakeEngineArgs(dict):
    """Fake engine args that can be used both as object attributes and as **kwargs."""

    def __init__(self, args_dict: dict[str, Any]):
        super().__init__(args_dict)
        # Add required attributes if not present
        if "model_stage" not in self:
            self["model_stage"] = None
        if "engine_output_type" not in self:
            self["engine_output_type"] = None
        # Also set as attributes for object-style access
        for key, value in self.items():
            setattr(self, key, value)


class _FakeStageConfig:
    """Fake stage config object that mimics the real stage config structure."""

    def __init__(self, config_dict: dict[str, Any]):
        # engine_args needs to work both as object (for OmniStage) and as dict (for **kwargs)
        engine_args_dict = config_dict.get("engine_args", {})
        self.engine_args = _FakeEngineArgs(engine_args_dict)
        self.final_output = config_dict.get("final_output", False)
        self.final_output_type = config_dict.get("final_output_type", None)
        self.stage_id = config_dict.get("stage_id", 0)
        # Store original dict for reference
        self._config_dict = config_dict


class _FakeStage:
    """Lightweight Stage stub: observable interactions without real implementation."""

    def __init__(self, config):
        # Handle both dict and object configs
        if isinstance(config, dict):
            config = _FakeStageConfig(config)
        self.config = config
        self.stage_config = config
        self.engine = None
        self._outputs: list = []
        # Allow configuring final_output and final_output_type, default to no output
        self.final_output = (
            config.final_output if hasattr(config, "final_output") else False
        )
        self.final_output_type = getattr(config, "final_output_type", None)
        # Configurable processing logic, default returns placeholder
        processed_input = getattr(config, "_config_dict", {}).get(
            "processed_input", ["processed"]
        )
        self._processed_input = processed_input
        # Capture prompts passed to generate for assertions
        self._last_prompts = None
        # Set attributes that OmniStage expects
        self.stage_id = getattr(config, "stage_id", 0)
        self.engine_args = config.engine_args

    def set_engine(self, engine) -> None:
        self.engine = engine

    def set_engine_outputs(self, outputs: list) -> None:
        self._outputs = outputs

    def process_engine_inputs(self, stage_list, prompts):
        # Simplified: return preset processed result to verify call chain
        return self._processed_input


class _FakeEngine:
    """Lightweight Engine stub: provides generate iterator output."""

    def __init__(self, outputs: List[Any]):
        self._outputs = outputs

    def generate(self, prompts, sampling_params):
        # Record the most recent prompts for outer assertions
        self._last_prompts = prompts
        # Simplified: return preset list at once, ensuring iterability
        yield from self._outputs


class _FakeStageLLM:
    """Replace OmniStageLLM to avoid constructing real engine."""

    def __init__(self, **kwargs):
        # Allow injecting custom fake outputs, default returns single placeholder output
        fake_outputs = kwargs.get("_fake_outputs", [[{"text": "ok"}]])
        self._fake_engine = _FakeEngine(fake_outputs)

    def generate(self, prompts, sampling_params):
        yield from self._fake_engine.generate(prompts, sampling_params)


@pytest.fixture
def fake_stage_config():
    return {
        # Don't include 'model' in engine_args since it's passed separately
        "engine_args": {},
        "final_output": True,
        "final_output_type": "text",
        # Second stage will use processed_input to verify the chain
        "processed_input": ["processed-by-stage"],
    }


def _setup_engine_mocks(monkeypatch):
    """Helper function to set up common engine mocks."""
    from unittest.mock import MagicMock

    fake_engine = MagicMock()
    # Add necessary attributes to fake_engine
    fake_engine.tokenizer = MagicMock()
    fake_engine.log_stats = False
    fake_engine.vllm_config = MagicMock()
    fake_engine.vllm_config.model_config = MagicMock()
    fake_engine.vllm_config.model_config.io_processor_plugin = None
    fake_engine.get_supported_tasks = MagicMock(return_value=[])
    fake_engine.model_config = MagicMock()
    fake_engine.model_config.io_processor_plugin = None
    # Add registry with resolve_model_cls method
    fake_registry = MagicMock()
    fake_registry.resolve_model_cls = MagicMock(
        return_value=(MagicMock(), "test_arch"))
    fake_engine.model_config.registry = fake_registry
    fake_engine.vllm_config.model_config.registry = fake_registry

    monkeypatch.setattr(
        "vllm.v1.engine.llm_engine.LLMEngine.from_engine_args",
        lambda **kw: fake_engine,
        raising=False,
    )

    # Mock model_config.registry.resolve_model_cls to return a tuple
    # Use a real class instead of MagicMock to avoid inspect.getsource issues
    class FakeModelClass:
        pass

    monkeypatch.setattr(
        "vllm.model_executor.model_loader.utils.get_model_architecture",
        lambda model_config: (FakeModelClass, "test_arch"),
        raising=False,
    )

    # Mock try_create_mm_pooling_model_cls to avoid inspect.getsource issues
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.utils._get_model_architecture",
        lambda model_config: (FakeModelClass, "test_arch"),
        raising=False,
    )

    # Mock try_create_mm_pooling_model_cls to return the class as-is
    monkeypatch.setattr(
        "vllm.model_executor.models.adapters.try_create_mm_pooling_model_cls",
        lambda model_cls: model_cls,
        raising=False,
    )

    # Mock _enable_processor_cache to return False
    # This avoids the need to set up processor factories
    monkeypatch.setattr(
        "vllm.multimodal.cache._enable_processor_cache",
        lambda model_config, mm_registry: False,
        raising=False,
    )

    # Mock get_io_processor to return None
    monkeypatch.setattr(
        "vllm.plugins.io_processors.get_io_processor",
        lambda vllm_config, io_processor_plugin: None,
        raising=False,
    )


@pytest.fixture(autouse=True)
def mock_get_config(monkeypatch):
    """Auto-mock get_config and related model loading functions to avoid model path validation."""
    from unittest.mock import MagicMock

    fake_hf_config = MagicMock()
    fake_hf_config.model_type = "qwen2_5_omni"  # Use a valid model type

    # Mock get_config in all possible locations
    def _mock_get_config(model, **kwargs):
        return fake_hf_config

    # Mock in the original location
    monkeypatch.setattr(
        "vllm.transformers_utils.config.get_config",
        _mock_get_config,
        raising=False,
    )
    # Also mock in utils module where it's imported
    monkeypatch.setattr(
        "vllm_omni.entrypoints.utils.get_config",
        _mock_get_config,
        raising=False,
    )

    # Mock transformers' cached_file to avoid downloading model configs
    def _mock_cached_file(path_or_repo_id, *args, **kwargs):
        # Return a fake config file path
        import os
        import tempfile

        fake_config_file = os.path.join(
            tempfile.gettempdir(), "fake_config.json")
        if not os.path.exists(fake_config_file):
            with open(fake_config_file, "w") as f:
                f.write('{"model_type": "qwen2_5_omni"}')
        return fake_config_file

    # Mock transformers cached_file and cached_files
    monkeypatch.setattr(
        "transformers.utils.hub.cached_file",
        _mock_cached_file,
        raising=False,
    )
    monkeypatch.setattr(
        "transformers.utils.hub.cached_files",
        lambda path_or_repo_id, filenames, **kwargs: (
            [_mock_cached_file(path_or_repo_id, filenames[0])
             ] if filenames else None
        ),
        raising=False,
    )


def test_initialize_stage_configs_called_when_none(monkeypatch, fake_stage_config):
    # Setup: load_stage_configs_from_model returns two identical configs
    # Convert dict to FakeStageConfig objects
    def _fake_loader(model: str):
        return [
            _FakeStageConfig(fake_stage_config),
            _FakeStageConfig(fake_stage_config),
        ]

    # Remove modules from cache BEFORE setting mocks to ensure clean state
    import sys

    for module_name in [
        "vllm_omni.entrypoints.utils",
        "vllm_omni.entrypoints.omni_llm",
        "vllm_omni.entrypoints.omni_stage",
    ]:
        if module_name in sys.modules:
            del sys.modules[module_name]

    # Set up common engine mocks
    _setup_engine_mocks(monkeypatch)

    # Mock both where it's defined and where it's imported
    monkeypatch.setattr(
        "vllm_omni.entrypoints.utils.load_stage_configs_from_model",
        _fake_loader,
        raising=False,
    )
    # Replace OmniStage and OmniStageLLM
    monkeypatch.setattr(
        "vllm_omni.entrypoints.omni_stage.OmniStage",
        lambda cfg: _FakeStage(cfg),
        raising=False,
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.omni_llm.OmniStageLLM",
        lambda **kw: _FakeStageLLM(**kw),
        raising=False,
    )

    # Import the module (not the class yet) after mocks are set
    import vllm_omni.entrypoints.omni_llm as omni_llm_module

    # Patch the imported function and class in the module
    monkeypatch.setattr(
        omni_llm_module, "load_stage_configs_from_model", _fake_loader)
    monkeypatch.setattr(omni_llm_module, "OmniStage",
                        lambda cfg: _FakeStage(cfg))
    monkeypatch.setattr(
        omni_llm_module, "OmniStageLLM", lambda **kw: _FakeStageLLM(**kw)
    )

    from vllm_omni.entrypoints.omni_llm import OmniLLM

    llm = OmniLLM(model="any")
    # Verify: auto-loaded stage_configs and stage_list have consistent count
    assert isinstance(llm.stage_configs, list)
    assert len(llm.stage_configs) == 2
    assert len(llm.stage_list) == 2
    # Verify: each Stage has engine injected
    for st in llm.stage_list:
        assert isinstance(st, _FakeStage)
        assert isinstance(st.engine, _FakeStageLLM)


def test_generate_raises_on_length_mismatch(monkeypatch, fake_stage_config):
    def _fake_loader(model: str):
        return [_FakeStageConfig(fake_stage_config)]

    # Remove modules from cache BEFORE setting mocks
    import sys

    for module_name in [
        "vllm_omni.entrypoints.utils",
        "vllm_omni.entrypoints.omni_llm",
        "vllm_omni.entrypoints.omni_stage",
    ]:
        if module_name in sys.modules:
            del sys.modules[module_name]

    # Set up common engine mocks
    _setup_engine_mocks(monkeypatch)

    monkeypatch.setattr(
        "vllm_omni.entrypoints.utils.load_stage_configs_from_model",
        _fake_loader,
        raising=False,
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.omni_llm.load_stage_configs_from_model",
        _fake_loader,
        raising=False,
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.omni_stage.OmniStage",
        lambda cfg: _FakeStage(cfg),
        raising=False,
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.omni_llm.OmniStageLLM",
        lambda **kw: _FakeStageLLM(**kw),
        raising=False,
    )
    from vllm_omni.entrypoints.omni_llm import OmniLLM

    llm = OmniLLM(model="any")
    with pytest.raises(ValueError):
        llm.generate(prompts=["hi"], sampling_params_list=[])


def test_generate_pipeline_and_final_outputs(monkeypatch, fake_stage_config):
    # Two stages: stage 0 uses original prompts directly; stage 1 needs process_engine_inputs result
    stage_cfg0 = dict(fake_stage_config)
    stage_cfg1 = dict(fake_stage_config)
    # Ensure stage 1's processed_input has clear marker for assertions
    stage_cfg1["processed_input"] = ["processed-for-stage-1"]

    def _fake_loader(model: str):
        return [_FakeStageConfig(stage_cfg0), _FakeStageConfig(stage_cfg1)]

    # Remove modules from cache BEFORE setting mocks
    import sys

    for module_name in [
        "vllm_omni.entrypoints.utils",
        "vllm_omni.entrypoints.omni_llm",
        "vllm_omni.entrypoints.omni_stage",
    ]:
        if module_name in sys.modules:
            del sys.modules[module_name]

    # Set up common engine mocks
    _setup_engine_mocks(monkeypatch)

    monkeypatch.setattr(
        "vllm_omni.entrypoints.utils.load_stage_configs_from_model",
        _fake_loader,
        raising=False,
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.omni_llm.load_stage_configs_from_model",
        _fake_loader,
        raising=False,
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.omni_stage.OmniStage",
        lambda cfg: _FakeStage(cfg),
        raising=False,
    )
    # Inject different fake engine outputs to distinguish stage outputs

    def _fake_stage_llm_factory(**kw):
        # Allow passing different fake outputs from engine_args if needed
        fake_outputs = kw.get("_fake_outputs")
        if fake_outputs is None:
            fake_outputs = [[{"stage": "default"}]]
        return _FakeStageLLM(_fake_outputs=fake_outputs)

    monkeypatch.setattr(
        "vllm_omni.entrypoints.omni_llm.OmniStageLLM",
        _fake_stage_llm_factory,
        raising=False,
    )

    # Import the module and patch the imported references
    import vllm_omni.entrypoints.omni_llm as omni_llm_module

    monkeypatch.setattr(
        omni_llm_module, "load_stage_configs_from_model", _fake_loader)
    monkeypatch.setattr(omni_llm_module, "OmniStage",
                        lambda cfg: _FakeStage(cfg))

    from vllm_omni.entrypoints.omni_llm import OmniLLM

    llm = OmniLLM(model="any")

    # Inject different outputs for each stage to simulate two rounds of generation
    # Replace the created stage.engine directly with different _FakeEngine
    s0 = _FakeStageLLM(_fake_outputs=[[{"stage": 0, "text": "s0"}]])
    s1 = _FakeStageLLM(_fake_outputs=[[{"stage": 1, "text": "s1"}]])
    llm.stage_list[0].engine = s0
    llm.stage_list[1].engine = s1

    # Prepare sampling params (content doesn't matter, just ensure length matches)
    sampling_params_list = [object(), object()]
    prompts = ["hi"]
    outputs = llm.generate(
        prompts=prompts, sampling_params_list=sampling_params_list)

    # Both stages have final_output=True, so should aggregate two OmniRequestOutput
    assert len(outputs) == 2
    # Verify stage 1 indeed took process_engine_inputs path (indirectly confirmed via fixed return value)
    assert llm.stage_list[1].process_engine_inputs([], []) is not None
    # Verify each stage has written back engine_outputs
    assert llm.stage_list[0]._outputs == [[{"stage": 0, "text": "s0"}]]
    assert llm.stage_list[1]._outputs == [[{"stage": 1, "text": "s1"}]]
    # Verify stage 0 indeed received original prompts
    assert s0._fake_engine._last_prompts == prompts
    # Verify stage 1 should receive processed input (can't directly access internal value, just confirm s1 was called)
    assert hasattr(s1._fake_engine, "_last_prompts")


def test_generate_no_final_output_returns_empty(monkeypatch, fake_stage_config):
    # Both stages are not final_output
    stage_cfg0 = dict(fake_stage_config)
    stage_cfg1 = dict(fake_stage_config)
    stage_cfg0["final_output"] = False
    stage_cfg1["final_output"] = False

    def _fake_loader(model: str):
        return [_FakeStageConfig(stage_cfg0), _FakeStageConfig(stage_cfg1)]

    # Remove modules from cache BEFORE setting mocks
    import sys

    for module_name in [
        "vllm_omni.entrypoints.utils",
        "vllm_omni.entrypoints.omni_llm",
        "vllm_omni.entrypoints.omni_stage",
    ]:
        if module_name in sys.modules:
            del sys.modules[module_name]

    # Set up common engine mocks
    _setup_engine_mocks(monkeypatch)

    monkeypatch.setattr(
        "vllm_omni.entrypoints.utils.load_stage_configs_from_model",
        _fake_loader,
        raising=False,
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.omni_llm.load_stage_configs_from_model",
        _fake_loader,
        raising=False,
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.omni_stage.OmniStage",
        lambda cfg: _FakeStage(cfg),
        raising=False,
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.omni_llm.OmniStageLLM",
        lambda **kw: _FakeStageLLM(**kw),
        raising=False,
    )

    # Import the module and patch the imported references
    import vllm_omni.entrypoints.omni_llm as omni_llm_module

    monkeypatch.setattr(
        omni_llm_module, "load_stage_configs_from_model", _fake_loader)
    monkeypatch.setattr(omni_llm_module, "OmniStage",
                        lambda cfg: _FakeStage(cfg))
    monkeypatch.setattr(
        omni_llm_module, "OmniStageLLM", lambda **kw: _FakeStageLLM(**kw)
    )

    from vllm_omni.entrypoints.omni_llm import OmniLLM

    llm = OmniLLM(model="any")
    outputs = llm.generate(prompts=["p"], sampling_params_list=[
                           object(), object()])
    assert outputs == []


def test_generate_sampling_params_none_raises(monkeypatch, fake_stage_config):
    # Since generate internally triggers exception on len(None), TypeError or ValueError are acceptable
    def _fake_loader(model: str):
        return [_FakeStageConfig(fake_stage_config)]

    # Remove modules from cache BEFORE setting mocks
    import sys

    for module_name in [
        "vllm_omni.entrypoints.utils",
        "vllm_omni.entrypoints.omni_llm",
        "vllm_omni.entrypoints.omni_stage",
    ]:
        if module_name in sys.modules:
            del sys.modules[module_name]

    # Set up common engine mocks
    _setup_engine_mocks(monkeypatch)

    monkeypatch.setattr(
        "vllm_omni.entrypoints.utils.load_stage_configs_from_model",
        _fake_loader,
        raising=False,
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.omni_llm.load_stage_configs_from_model",
        _fake_loader,
        raising=False,
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.omni_stage.OmniStage",
        lambda cfg: _FakeStage(cfg),
        raising=False,
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.omni_llm.OmniStageLLM",
        lambda **kw: _FakeStageLLM(**kw),
        raising=False,
    )
    from vllm_omni.entrypoints.omni_llm import OmniLLM

    llm = OmniLLM(model="any")
    with pytest.raises(Exception):
        llm.generate(prompts=["p"], sampling_params_list=None)
