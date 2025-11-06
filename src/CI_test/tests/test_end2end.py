import os
import sys
import types
from types import SimpleNamespace

import pytest
import torch


def _install_fake_omni_llm_module(fake_generate):
    """Install a fake module vllm_omni.entrypoints.omni_llm with OmniLLM.

    This avoids importing real package and weights. The fake OmniLLM records
    calls and returns minimal outputs needed by end2end.py control flow.
    """
    pkg_root = types.ModuleType("vllm_omni")
    entrypoints = types.ModuleType("vllm_omni.entrypoints")
    omni_llm_mod = types.ModuleType("vllm_omni.entrypoints.omni_llm")

    class _FakeOmniLLM:
        def __init__(self, model: str):
            self.model = model
            self.calls = []

        def generate(self, prompts, sampling_params_list):
            self.calls.append((prompts, sampling_params_list))
            return fake_generate(prompts)

    omni_llm_mod.OmniLLM = _FakeOmniLLM

    sys.modules["vllm_omni"] = pkg_root
    sys.modules["vllm_omni.entrypoints"] = entrypoints
    sys.modules["vllm_omni.entrypoints.omni_llm"] = omni_llm_mod


def _load_end2end_module():
    """Load end2end.py by path to avoid package name issues (dash in dir)."""
    import importlib.util

    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../../vllm-omni/examples/offline_inference/qwen_2_5_omni/end2end.py",
        )
    )
    spec = importlib.util.spec_from_file_location("_end2end_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fake_stage_outputs(final_output_type: str, num_requests: int):
    reqs = []
    for i in range(num_requests):
        rid = f"req_{i}"
        if final_output_type == "text":
            reqs.append(
                SimpleNamespace(
                    request_id=rid,
                    outputs=[SimpleNamespace(text=f"text_{i}")],
                )
            )
        else:
            audio = torch.zeros(24000, dtype=torch.float32)
            reqs.append(
                SimpleNamespace(
                    request_id=rid,
                    multimodal_output={"audio": audio},
                )
            )
    return SimpleNamespace(final_output_type=final_output_type, request_output=reqs)


@pytest.fixture(autouse=True)
def _env_v1(monkeypatch):
    # Ensure V1 is set and cleaned per test
    monkeypatch.setenv("VLLM_USE_V1", "1")


def test_text_branch(tmp_path, monkeypatch, capsys):
    # Fake OmniLLM returning only text stage
    def fake_generate(prompts):
        return [_fake_stage_outputs("text", len(prompts))]

    _install_fake_omni_llm_module(fake_generate)

    # Fake soundfile.write should not be called in text branch
    calls = []

    def _fake_sf_write(path, array, samplerate):  # noqa: ARG001
        calls.append((path, samplerate))

    sys.modules["soundfile"] = types.SimpleNamespace(write=_fake_sf_write)

    mod = _load_end2end_module()

    argv = [
        "prog",
        "--model",
        "dummy",
        "--prompts",
        "hello",
    ]
    monkeypatch.setenv("PYTHONHASHSEED", "42")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("PYTHONWARNINGS", "ignore")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(sys, "argv", argv)

    mod.main()
    out = capsys.readouterr().out

    assert "Request ID:" in out
    assert "Text Output:" in out
    assert len(calls) == 0


def test_audio_branch(tmp_path, monkeypatch, capsys):
    # Fake OmniLLM returning only audio stage
    def fake_generate(prompts):
        return [_fake_stage_outputs("audio", len(prompts))]

    _install_fake_omni_llm_module(fake_generate)

    sf_calls = []

    def _fake_sf_write(path, array, samplerate):
        # minimal check, do not write to disk
        sf_calls.append((path, getattr(array, "shape", None), samplerate))

    sys.modules["soundfile"] = types.SimpleNamespace(write=_fake_sf_write)

    mod = _load_end2end_module()

    out_dir = tmp_path / "wav"
    argv = [
        "prog",
        "--model",
        "dummy",
        "--prompts",
        "hi",
        "--do-wave",
        "--output-wav",
        str(out_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    mod.main()
    out = capsys.readouterr().out

    # stdout contains save info, and fake write called once
    assert "Saved audio to" in out
    assert len(sf_calls) == 1
    path, shape, sr = sf_calls[0]
    assert path.endswith(".wav")
    assert sr == 24000
