# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import builtins
import hashlib
import importlib.util
import json
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from unit.mocks import store_message

from pyrit.models import ContentScorable, MessagePiece, ScoreStatus, ScoringExpectation
from pyrit.score import LayaRefusalScorer, MessageScorable
from pyrit.score.true_false import laya_refusal_scorer as module
from pyrit.score.true_false.laya_refusal_scorer import (
    _format_response,
    _LayaEncoder,
    _load_training_rows,
    _predict_probability,
    _train_head,
)

# The dev install has no torch; these tests fit the real head or build real tensors, so they run
# wherever torch is installed (the dev_all matrix and the huggingface extra).
requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="LayaRefusalScorer needs torch for its head"
)


def _separable_training_data(rows_per_class: int = 150) -> tuple[list[list[float]], list[int]]:
    """Two linearly separable clusters in a small feature space."""
    features: list[list[float]] = []
    labels: list[int] = []
    for index in range(rows_per_class):
        jitter = (index % 5) * 0.01
        features.append([4.0 + jitter, -2.0 - jitter, 0.1])
        features.append([-4.0 - jitter, 2.0 + jitter, 0.1])
        labels += [1, 0]
    return features, labels


def _trained_head() -> module._TrainedHead:
    features, labels = _separable_training_data()
    return _train_head(features=features, labels=labels)


def _scorer_with_mocks(
    *,
    features: list[float],
    abstain_band: tuple[float, float] | None = (0.2, 0.8),
) -> LayaRefusalScorer:
    scorer = LayaRefusalScorer(abstain_band=abstain_band)
    scorer._head = _trained_head()
    encoder = MagicMock(spec=_LayaEncoder)
    encoder.features_async = AsyncMock(return_value=[features])
    scorer._encoder = encoder
    return scorer


def test_format_response_truncates_to_the_trained_length():
    assert _format_response("abc") == "abc"
    assert len(_format_response("x" * 900)) == module._TRUNCATION_CHARS


def test_load_training_rows_matches_pinned_datasets():
    pairs, labels = _load_training_rows()

    assert len(pairs) == len(labels) == 169
    assert set(labels) == {0, 1}
    assert all(len(response) <= module._TRUNCATION_CHARS for _, response in pairs)


def test_load_training_rows_rejects_changed_dataset(monkeypatch, tmp_path):
    tampered = b"# dataset_version=1.0\nobjective,assistant_response,human_score,data_type\n"
    for file_name in module._TRAINING_DATASETS:
        (tmp_path / file_name).write_bytes(tampered)
    monkeypatch.setattr(module, "_REFUSAL_EVALS_PATH", tmp_path)

    with pytest.raises(RuntimeError, match="does not match the SHA-256"):
        _load_training_rows()


def test_compute_dataset_hashes_matches_pins():
    assert LayaRefusalScorer.compute_dataset_hashes() == module._TRAINING_DATASETS


@requires_torch
def test_train_head_is_deterministic_and_separates():
    first, second = _trained_head(), _trained_head()

    assert first == second
    assert first.training_rows == 300
    features, _ = _separable_training_data()
    assert _predict_probability(head=first, features=features[0]) > 0.9
    assert _predict_probability(head=first, features=features[1]) < 0.1


@requires_torch
def test_predict_probability_is_a_probability():
    head = _trained_head()

    assert 0.0 <= _predict_probability(head=head, features=[0.0, 0.0, 0.0]) <= 1.0


@pytest.mark.parametrize("band", [(0.8, 0.2), (-0.1, 0.5), (0.5, 1.5), (0.4, 0.4)])
def test_invalid_abstain_band_raises(band):
    with pytest.raises(ValueError, match="abstain_band"):
        LayaRefusalScorer(abstain_band=band)


@requires_torch
@pytest.mark.usefixtures("patch_central_database")
async def test_score_async_returns_refusal_outside_band():
    features, _ = _separable_training_data()
    scorer = _scorer_with_mocks(features=features[0])

    (score,) = await scorer.score_async(scorable=ContentScorable(value="I can't help with that."))

    assert score.status is ScoreStatus.COMPLETE
    assert score.get_value() is True
    assert score.score_category == ["refusal"]
    assert score.score_metadata["refusal_probability"] > 0.8


@requires_torch
@pytest.mark.usefixtures("patch_central_database")
async def test_score_async_returns_compliance_outside_band():
    features, _ = _separable_training_data()
    scorer = _scorer_with_mocks(features=features[1])

    (score,) = await scorer.score_async(scorable=ContentScorable(value="Here is how you bake bread."))

    assert score.status is ScoreStatus.COMPLETE
    assert score.get_value() is False
    assert score.score_metadata["refusal_probability"] < 0.2


@requires_torch
@pytest.mark.usefixtures("patch_central_database")
async def test_score_async_abstains_inside_band():
    # A feature vector between the two training clusters lands near 0.5.
    scorer = _scorer_with_mocks(features=[0.0, 0.0, 0.1], abstain_band=(0.1, 0.9))

    (score,) = await scorer.score_async(scorable=ContentScorable(value="Maybe."))

    assert score.status is ScoreStatus.UNDETERMINED
    assert score.score_value is None
    assert score.score_metadata["abstain_band_low"] == 0.1
    assert score.score_metadata["abstain_band_high"] == 0.9
    assert 0.1 <= score.score_metadata["refusal_probability"] <= 0.9


@requires_torch
@pytest.mark.usefixtures("patch_central_database")
async def test_score_async_with_band_disabled_never_abstains():
    scorer = _scorer_with_mocks(features=[0.0, 0.0, 0.1], abstain_band=None)

    (score,) = await scorer.score_async(scorable=ContentScorable(value="Maybe."))

    assert score.status is ScoreStatus.COMPLETE
    assert "abstain_band_low" not in score.score_metadata


@requires_torch
@pytest.mark.usefixtures("patch_central_database")
async def test_score_async_passes_objective_and_truncated_response():
    features, _ = _separable_training_data()
    scorer = _scorer_with_mocks(features=features[0])

    await scorer.score_async(
        scorable=ContentScorable(value="y" * 900),
        expectation=ScoringExpectation(objective="the objective"),
    )

    (pair,) = scorer._encoder.features_async.await_args.kwargs["texts"]
    assert pair[0] == "the objective"
    assert pair[1] == "y" * module._TRUNCATION_CHARS


@requires_torch
@pytest.mark.usefixtures("patch_central_database")
async def test_score_async_without_objective_passes_empty_string():
    features, _ = _separable_training_data()
    scorer = _scorer_with_mocks(features=features[0])

    await scorer.score_async(scorable=ContentScorable(value="no."))

    (pair,) = scorer._encoder.features_async.await_args.kwargs["texts"]
    assert pair[0] == ""


@requires_torch
@pytest.mark.usefixtures("patch_central_database")
async def test_load_model_async_trains_once(monkeypatch):
    scorer = LayaRefusalScorer()
    features, labels = _separable_training_data()
    monkeypatch.setattr(module, "_load_training_rows", lambda: ([("o", "r")] * len(labels), labels))
    encoder = MagicMock(spec=_LayaEncoder)
    encoder.features_async = AsyncMock(return_value=features)
    scorer._encoder = encoder

    await scorer.load_model_async()
    await scorer.load_model_async()

    assert encoder.features_async.await_count == 1
    assert scorer._head is not None
    assert scorer._head.training_rows == len(labels)


def test_identifier_contains_configuration():
    params = LayaRefusalScorer(abstain_band=(0.1, 0.9)).get_identifier().params

    assert params["abstain_band"] == [0.1, 0.9]
    assert params["model_id"] == _LayaEncoder.DEFAULT_MODEL_ID
    assert params["revision"] == _LayaEncoder.DEFAULT_MODEL_REVISION


def test_identifier_changes_with_band():
    first = LayaRefusalScorer(abstain_band=(0.1, 0.9)).get_identifier()
    second = LayaRefusalScorer(abstain_band=None).get_identifier()

    assert first.hash != second.hash


def test_encoder_reports_missing_package(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "laya":
            raise ModuleNotFoundError("No module named 'laya'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="pip install laya"):
        _LayaEncoder()._load_model()


def test_encoder_pins_revision_and_restricts_download(monkeypatch):
    fake_laya = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "laya", fake_laya)
    snapshot = MagicMock(return_value="/cache/laya")
    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)

    _LayaEncoder(device="cpu")._load_model()

    assert snapshot.call_args.kwargs["revision"] == _LayaEncoder.DEFAULT_MODEL_REVISION
    assert set(snapshot.call_args.kwargs["allow_patterns"]) == set(module._CHECKPOINT_FILES)
    fake_laya.load.assert_called_once_with("/cache/laya", device="cpu")


def test_encoder_uses_local_directory_without_download(monkeypatch, tmp_path):
    fake_laya = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "laya", fake_laya)
    snapshot = MagicMock()
    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)

    _LayaEncoder(model_id=str(tmp_path))._load_model()

    snapshot.assert_not_called()
    fake_laya.load.assert_called_once_with(str(tmp_path), device=None)


async def test_encoder_returns_empty_without_texts():
    assert await _LayaEncoder().features_async(texts=[]) == []


def test_question_presents_both_options():
    assert set(module._QUESTION["criteria"]) == {"refuses", "complies"}
    assert "`response`" in module._QUESTION["instructions"]
    assert "`objective`" in module._QUESTION["instructions"]


def test_pinned_datasets_exist_with_expected_hashes():
    for file_name, expected in module._TRAINING_DATASETS.items():
        raw = (module._REFUSAL_EVALS_PATH / file_name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected


def _fake_laya_common(monkeypatch, calls: list[list[int]], states: list[dict] | None = None):
    """Install a stand-in for laya.common that records the option orders it is asked for."""
    import sys
    import types

    import torch

    states = states if states is not None else []
    common = types.ModuleType("laya.common")
    common.QTYPES = {"choice": 0}

    def build_sequence(tokenizer, state, question, max_len, head_max_len, option_order=None):
        calls.append(list(option_order or [0, 1]))
        states.append(dict(state))
        return [1, 2, 3, 4], [1, 2]

    def collate_items(batch, pad_id):
        return {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
            "marker_pos": torch.tensor([[1, 2]]),
            "qtype": torch.tensor([0]),
        }

    common.build_sequence = build_sequence
    common.collate_items = collate_items
    monkeypatch.setitem(sys.modules, "laya", types.ModuleType("laya"))
    monkeypatch.setitem(sys.modules, "laya.common", common)


def _fake_agent():
    """An agent whose hidden states differ per option position, so order matters."""
    import torch

    agent = MagicMock()
    agent.cfg = {"max_len": 512, "head_max_len": 192}
    agent.device = torch.device("cpu")
    agent.tok.pad_token_id = 0
    hidden = torch.tensor([[[0.0, 0.0], [1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]])
    agent.model.encoder.return_value = MagicMock(last_hidden_state=hidden)
    agent.model.type_emb.return_value = torch.zeros((1, 2))
    agent.model.head.layers = []
    return agent


@requires_torch
async def test_features_average_both_option_orders(monkeypatch):
    calls: list[list[int]] = []
    _fake_laya_common(monkeypatch, calls)
    encoder = _LayaEncoder()
    encoder._agent = _fake_agent()

    (vector,) = await encoder.features_async(texts=[("objective", "response")])

    assert calls == [[0, 1], [1, 0]]
    # The stub returns the same hidden states either way, so restoring the canonical order
    # and averaging must give back the markers in canonical order rather than their mean.
    assert vector == [2.0, 3.0, 2.0, 3.0]


@requires_torch
async def test_features_put_the_response_before_a_bounded_objective(monkeypatch):
    # Laya keeps only the start of the serialized state, so an objective placed first could push
    # the response out of the encoded input entirely.
    calls: list[list[int]] = []
    states: list[dict] = []
    _fake_laya_common(monkeypatch, calls, states)
    encoder = _LayaEncoder()
    encoder._agent = _fake_agent()

    await encoder.features_async(texts=[("context " * 2000, "I cannot help with that.")])

    assert states
    for state in states:
        assert list(state) == ["response", "objective"]
        assert state["response"] == "I cannot help with that."
        assert len(state["objective"]) == module._OBJECTIVE_CHARS


def test_build_state_bounds_the_objective_and_keeps_the_response_first():
    state = module._build_state(objective="x" * 5000, response="no")

    assert list(state) == ["response", "objective"]
    assert state["response"] == "no"
    assert len(state["objective"]) == module._OBJECTIVE_CHARS
    assert module._build_state(objective="short", response="r")["objective"] == "short"


@requires_torch
def test_train_head_leaves_the_callers_random_state_alone():
    import torch

    torch.manual_seed(1234)
    before = torch.get_rng_state().clone()
    _trained_head()
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("data_type", ["text", "error"])
async def test_blocked_response_counts_as_a_refusal(data_type):
    scorer = LayaRefusalScorer()
    encoder = MagicMock(spec=_LayaEncoder)
    encoder.features_async = AsyncMock()
    scorer._encoder = encoder
    request = MessagePiece(
        role="assistant",
        original_value="blocked response",
        original_value_data_type=data_type,
        converted_value_data_type=data_type,
        response_error="blocked",
        conversation_id=str(uuid4()),
    ).to_message()

    (score,) = await scorer.score_async(scorable=MessageScorable.from_message(store_message(request)))

    assert score.status is ScoreStatus.COMPLETE
    assert score.get_value() is True
    assert score.score_category == ["refusal"]
    assert score.score_rationale == "Content was filtered, constituting a refusal."
    encoder.features_async.assert_not_awaited()


@pytest.mark.usefixtures("patch_central_database")
async def test_other_response_errors_stay_undetermined():
    scorer = LayaRefusalScorer()
    encoder = MagicMock(spec=_LayaEncoder)
    encoder.features_async = AsyncMock()
    scorer._encoder = encoder
    request = MessagePiece(
        role="assistant",
        original_value="upstream failure",
        original_value_data_type="error",
        converted_value_data_type="error",
        response_error="unknown",
        conversation_id=str(uuid4()),
    ).to_message()

    (score,) = await scorer.score_async(scorable=MessageScorable.from_message(store_message(request)))

    assert score.status is ScoreStatus.UNDETERMINED
    encoder.features_async.assert_not_awaited()


async def test_default_evaluation_requires_an_explicit_mapping():
    scorer = LayaRefusalScorer()

    assert scorer.evaluation_file_mapping is None
    with pytest.raises(ValueError, match="No file_mapping provided"):
        await scorer.evaluate_async()


def test_identifier_carries_the_training_recipe():
    params = LayaRefusalScorer().get_identifier().params

    assert params["recipe_version"] == module._RECIPE["version"]
    expected = hashlib.sha256(json.dumps(module._RECIPE, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    assert params["recipe_digest"] == expected


def test_recipe_covers_everything_that_shapes_the_head():
    recipe = module._RECIPE

    assert recipe["datasets"] == module._TRAINING_DATASETS
    assert recipe["question"] == module._QUESTION
    assert recipe["response_chars"] == module._TRUNCATION_CHARS
    assert recipe["objective_chars"] == module._OBJECTIVE_CHARS
    assert recipe["state_order"] == ["response", "objective"]
    assert recipe["head"]["weight_decay"] == module._WEIGHT_DECAY
