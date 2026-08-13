"""Proves TinkerBackend is wired to the documented Tinker SDK call sequence
(create_lora_training_client -> forward_backward_async/optim_step_async loop ->
save_weights_and_get_sampling_client) without needing a real TINKER_API_KEY,
using a mocked `tinker` module. Also proves the backend refuses to silently
no-op when no key is configured."""

import sys
import types
from unittest.mock import MagicMock

import pytest

from app.backends.base import BackendUnavailableError
from app.models.run import DatasetSpec, EvalSpec, HyperParams, RunPlan


def _make_fake_tinker_module():
    fake = types.ModuleType("tinker")

    fb_future = MagicMock()
    fb_future.result.return_value = MagicMock(loss=1.23)
    optim_future = MagicMock()
    optim_future.result.return_value = MagicMock()

    training_client = MagicMock()
    training_client.get_tokenizer.return_value = MagicMock(
        encode=lambda text, add_special_tokens=False: [1, 2, 3],
        decode=lambda ids: "decoded text",
    )
    training_client.forward_backward_async.return_value = fb_future
    training_client.optim_step_async.return_value = optim_future

    sample_future = MagicMock()
    sampled_seq = MagicMock(tokens=[4, 5, 6])
    sample_future.result.return_value = MagicMock(sequences=[sampled_seq])
    sampling_client = MagicMock()
    sampling_client.sample_async.return_value = sample_future
    training_client.save_weights_and_get_sampling_client.return_value = sampling_client

    service_client = MagicMock()
    service_client.create_lora_training_client.return_value = training_client

    fake.ServiceClient = MagicMock(return_value=service_client)
    fake.Datum = lambda model_input, loss_fn_inputs: {"model_input": model_input, "loss_fn_inputs": loss_fn_inputs}
    fake.ModelInput = MagicMock()
    fake.ModelInput.from_ints = lambda tokens: {"tokens": tokens}
    fake.TensorData = lambda data: {"data": data}
    fake.AdamParams = lambda learning_rate: {"learning_rate": learning_rate}
    fake.SamplingParams = lambda max_tokens: {"max_tokens": max_tokens}
    return fake, service_client, training_client


def _plan():
    return RunPlan(
        task_prompt="test",
        technique="lora_sft",
        base_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        backend="tinker",
        dataset_spec=DatasetSpec(source_type="synthetic", n_train=2, n_eval_holdout=1),
        hyperparams=HyperParams(max_steps=2, per_device_batch_size=1),
        eval_spec=EvalSpec(evaluator="perplexity"),
    )


def test_raises_when_no_api_key(monkeypatch):
    from app.config import get_settings

    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    get_settings.cache_clear()

    from app.backends.tinker_backend import TinkerBackend

    with pytest.raises(BackendUnavailableError):
        TinkerBackend()
    get_settings.cache_clear()


def test_call_sequence_with_mocked_sdk(monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("TINKER_API_KEY", "fake-key-for-test")
    get_settings.cache_clear()

    fake_tinker, service_client, training_client = _make_fake_tinker_module()
    monkeypatch.setitem(sys.modules, "tinker", fake_tinker)

    from app.backends.tinker_backend import TinkerBackend

    backend = TinkerBackend()

    dataset = [{"prompt": "hi", "completion": "there"}, {"prompt": "a", "completion": "b"}]
    handle = backend.prepare(_plan(), dataset)
    service_client.create_lora_training_client.assert_called_once()

    result = backend.run(handle, on_log=lambda msg, data: None)
    assert training_client.forward_backward_async.called
    assert training_client.optim_step_async.called
    assert training_client.save_weights_and_get_sampling_client.called
    assert len(result.loss_history) >= 1

    outputs = backend.sample(handle, ["hello"])
    assert outputs == ["decoded text"]

    get_settings.cache_clear()


def test_abliteration_not_supported_on_tinker(monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("TINKER_API_KEY", "fake-key-for-test")
    get_settings.cache_clear()

    fake_tinker, _, _ = _make_fake_tinker_module()
    monkeypatch.setitem(sys.modules, "tinker", fake_tinker)

    from app.backends.tinker_backend import TinkerBackend

    backend = TinkerBackend()
    plan = _plan()
    plan.technique = "abliteration"
    with pytest.raises(NotImplementedError):
        backend.prepare(plan, [])

    get_settings.cache_clear()
