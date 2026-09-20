# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.memory import CentralMemory, MemoryInterface
from pyrit.models import ComponentIdentifier, Score, ScoreStatus
from pyrit.score import FloatScaleFallbackScorer, SubStringScorer
from pyrit.score.float_scale.float_scale_fallback_scorer import PRIMARY_RATIONALE_KEY, RESOLVED_BY_KEY
from pyrit.score.float_scale.float_scale_scorer import MessageFloatScaleScorer


def _score(*, value: float | None, class_name: str, rationale: str = "", metadata: dict | None = None) -> Score:
    return Score(
        score_value=None if value is None else str(value),
        score_type="float_scale",
        status=ScoreStatus.UNDETERMINED if value is None else ScoreStatus.COMPLETE,
        score_category=["violence"],
        score_rationale=rationale,
        score_metadata=metadata,
        message_piece_id=uuid.uuid4(),
        score_value_description=f"from {class_name}",
        scorer_class_identifier=ComponentIdentifier(class_name=class_name, class_module="test.mock"),
        id=uuid.uuid4(),
    )


def _mock_scorer(*, class_name: str, scores: list[Score] | None) -> MagicMock:
    """A float scale scorer that returns fixed scores. ``scores=None`` means non-applicable (``[]``)."""
    scorer = MagicMock(spec=MessageFloatScaleScorer)
    scorer._score_nested_async = AsyncMock(return_value=[] if scores is None else scores)
    scorer.get_identifier = MagicMock(return_value=ComponentIdentifier(class_name=class_name, class_module="test.mock"))
    scorer.get_chat_target = MagicMock(return_value=None)
    return scorer


@pytest.fixture
def memory():
    memory = MagicMock(MemoryInterface)
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        yield memory


async def test_primary_verdict_is_returned_without_consulting_the_fallback(memory):
    primary = _mock_scorer(class_name="Classifier", scores=[_score(value=0.9, class_name="Classifier")])
    fallback = _mock_scorer(class_name="Judge", scores=[_score(value=0.1, class_name="Judge")])
    scorer = FloatScaleFallbackScorer(scorer=primary, fallback_scorer=fallback)

    scores = await scorer.score_text_async(text="text")

    assert len(scores) == 1
    assert scores[0].get_value() == 0.9
    assert scores[0].score_metadata[RESOLVED_BY_KEY] == "primary"
    assert scores[0].scorer_class_identifier == scorer.get_identifier()
    fallback._score_nested_async.assert_not_called()


async def test_primary_abstention_is_routed_to_the_fallback(memory):
    primary = _mock_scorer(
        class_name="Classifier",
        scores=[
            _score(
                value=None,
                class_name="Classifier",
                rationale="inside the abstain band",
                metadata={"abstain_band_low": 0.3, "abstain_band_high": 0.7, "probability": 0.52},
            )
        ],
    )
    fallback = _mock_scorer(
        class_name="Judge", scores=[_score(value=0.8, class_name="Judge", rationale="clearly violent")]
    )
    scorer = FloatScaleFallbackScorer(scorer=primary, fallback_scorer=fallback)

    scores = await scorer.score_text_async(text="text")

    assert len(scores) == 1
    score = scores[0]
    assert not score.is_undetermined
    assert score.get_value() == 0.8
    assert score.score_metadata[RESOLVED_BY_KEY] == "fallback"
    # Nothing the primary said is lost.
    assert score.score_metadata[PRIMARY_RATIONALE_KEY] == "inside the abstain band"
    assert score.score_metadata["abstain_band_low"] == 0.3
    assert score.score_metadata["probability"] == 0.52
    assert score.score_rationale.startswith(
        "Classifier abstained, so the score was routed to Judge, which returned 0.8."
    )
    assert "clearly violent" in score.score_rationale
    assert score.scorer_class_identifier == scorer.get_identifier()
    fallback._score_nested_async.assert_awaited_once()


async def test_double_abstention_stays_undetermined_and_says_so(memory):
    primary = _mock_scorer(class_name="Classifier", scores=[_score(value=None, class_name="Classifier")])
    fallback = _mock_scorer(class_name="Judge", scores=[_score(value=None, class_name="Judge", rationale="unsure")])
    scorer = FloatScaleFallbackScorer(scorer=primary, fallback_scorer=fallback)

    scores = await scorer.score_text_async(text="text")

    assert len(scores) == 1
    assert scores[0].is_undetermined
    assert scores[0].score_metadata[RESOLVED_BY_KEY] == "fallback"
    assert "which also abstained" in scores[0].score_rationale


async def test_fallback_keys_win_over_primary_keys_on_a_clash(memory):
    primary = _mock_scorer(
        class_name="Classifier", scores=[_score(value=None, class_name="Classifier", metadata={"model": "cpu"})]
    )
    fallback = _mock_scorer(
        class_name="Judge", scores=[_score(value=0.4, class_name="Judge", metadata={"model": "llm"})]
    )
    scorer = FloatScaleFallbackScorer(scorer=primary, fallback_scorer=fallback)

    scores = await scorer.score_text_async(text="text")

    assert scores[0].score_metadata["model"] == "llm"


async def test_non_applicable_primary_is_propagated_silently(memory):
    primary = _mock_scorer(class_name="Classifier", scores=None)
    fallback = _mock_scorer(class_name="Judge", scores=[_score(value=0.8, class_name="Judge")])
    scorer = FloatScaleFallbackScorer(scorer=primary, fallback_scorer=fallback)

    scores = await scorer.score_text_async(text="text")

    assert scores == []
    fallback._score_nested_async.assert_not_called()


async def test_non_applicable_fallback_leaves_the_primary_abstention_standing(memory):
    primary = _mock_scorer(class_name="Classifier", scores=[_score(value=None, class_name="Classifier")])
    fallback = _mock_scorer(class_name="Judge", scores=None)
    scorer = FloatScaleFallbackScorer(scorer=primary, fallback_scorer=fallback)

    scores = await scorer.score_text_async(text="text")

    assert len(scores) == 1
    assert scores[0].is_undetermined
    assert scores[0].score_metadata[RESOLVED_BY_KEY] == "primary"


async def test_fallback_error_is_not_swallowed(memory):
    primary = _mock_scorer(class_name="Classifier", scores=[_score(value=None, class_name="Classifier")])
    fallback = _mock_scorer(class_name="Judge", scores=None)
    fallback._score_nested_async = AsyncMock(side_effect=RuntimeError("judge unavailable"))
    scorer = FloatScaleFallbackScorer(scorer=primary, fallback_scorer=fallback)

    with pytest.raises(RuntimeError, match="judge unavailable"):
        await scorer.score_text_async(text="text")


def test_identifier_names_both_sub_scorers(memory):
    primary = _mock_scorer(class_name="Classifier", scores=[])
    fallback = _mock_scorer(class_name="Judge", scores=[])
    scorer = FloatScaleFallbackScorer(scorer=primary, fallback_scorer=fallback)

    identifier = scorer.get_identifier()

    assert identifier.class_name == "FloatScaleFallbackScorer"
    assert [s.class_name for s in identifier.children["sub_scorers"]] == ["Classifier", "Judge"]


def test_rejects_a_true_false_scorer_and_the_same_scorer_twice(memory):
    float_scorer = _mock_scorer(class_name="Classifier", scores=[])
    true_false = SubStringScorer(substring="x", categories=["c"])

    with pytest.raises(ValueError, match="must be a float scale scorer"):
        FloatScaleFallbackScorer(scorer=true_false, fallback_scorer=float_scorer)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fallback scorer must be a float scale scorer"):
        FloatScaleFallbackScorer(scorer=float_scorer, fallback_scorer=true_false)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="different scorer"):
        FloatScaleFallbackScorer(scorer=float_scorer, fallback_scorer=float_scorer)


def test_chat_target_falls_through_to_the_fallback(memory):
    primary = _mock_scorer(class_name="Classifier", scores=[])
    fallback = _mock_scorer(class_name="Judge", scores=[])
    fallback.get_chat_target = MagicMock(return_value="judge-target")
    scorer = FloatScaleFallbackScorer(scorer=primary, fallback_scorer=fallback)

    assert scorer.get_chat_target() == "judge-target"
