# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import dataclasses
from typing import Literal, cast
from unittest.mock import MagicMock

import pytest

from pyrit.models import (
    ComponentIdentifier,
    Condition,
    ContentEntryScorable,
    ContentScorable,
    MatchesObjective,
    Message,
    MessagePiece,
    MessageScorable,
    ProgramBehaves,
    ProgramExecutionScorable,
    Scorable,
    ScorableUnion,
    Score,
    ScoringExpectation,
    ToolsCalled,
)
from pyrit.score import (
    LegacyScorerHookError,
    ManualScorer,
    MessageToolCallScorer,
    MessageTrueFalseScorer,
    ProgramExecutionScorer,
    ScorerCapabilities,
    ScorerPromptValidator,
    SelfAskGeneralFloatScaleScorer,
    SelfAskScaleScorer,
    TrueFalseCompositeScorer,
    TrueFalseInverterScorer,
    TrueFalseScoreAggregator,
    TrueFalseScorer,
    create_conversation_scorer,
)
from pyrit.score.audio_transcript_scorer import AudioTranscriptHelper
from pyrit.score.float_scale.decision_scorer import DecisionScorer
from pyrit.score.observation import execution
from pyrit.score.video_scorer import VideoHelper

pytestmark = pytest.mark.usefixtures("patch_central_database")


class _TypedCondition(Condition):
    condition_type: Literal["test_capabilities_typed"] = "test_capabilities_typed"
    value: str = "typed"


def _score_for(scorer: TrueFalseScorer, piece: MessagePiece, expectation: ScoringExpectation | None) -> list[Score]:
    return [
        Score(
            score_value="true",
            score_type="true_false",
            message_piece_id=piece.id,
            scored_expectation=expectation,
            scorer_class_identifier=scorer.get_identifier(),
        )
    ]


class _TypedMessageScorer(MessageTrueFalseScorer):
    CONDITION_TYPE = _TypedCondition

    def __init__(self, *, validator: ScorerPromptValidator | None = None) -> None:
        super().__init__(validator=validator or ScorerPromptValidator())

    def _build_identifier(self) -> ComponentIdentifier:
        return self._create_identifier()

    async def _score_piece_with_expectation_async(
        self, message_piece: MessagePiece, *, expectation: ScoringExpectation | None
    ) -> list[Score]:
        return _score_for(self, message_piece, expectation)


class _UndeclaredMessageScorer(_TypedMessageScorer):
    CONDITION_TYPE = None


class _LegacyPieceScorer(MessageTrueFalseScorer):
    """A pre-migration leaf: only the objective-only piece hook."""

    def __init__(self) -> None:
        super().__init__(validator=ScorerPromptValidator())

    def _build_identifier(self) -> ComponentIdentifier:
        return self._create_identifier()

    async def _score_piece_async(self, message_piece: MessagePiece, *, objective: str | None = None) -> list[Score]:
        return _score_for(self, message_piece, None)


class _GenericScorer(TrueFalseScorer):
    def _build_identifier(self) -> ComponentIdentifier:
        return self._create_identifier()

    async def _score_scorable_async(self, *, scorable: Scorable, expectation: ScoringExpectation | None) -> list[Score]:
        return [
            Score(
                score_value="true",
                score_type="true_false",
                scorable=cast("ScorableUnion", scorable),
                scorer_class_identifier=self.get_identifier(),
            )
        ]


class _LegacyDirectScorer(TrueFalseScorer):
    def __init__(self) -> None:
        super().__init__(validator=ScorerPromptValidator(supported_data_types=["audio_path"]))

    def _build_identifier(self) -> ComponentIdentifier:
        return self._create_identifier()

    async def _score_async(self, message: Message, *, objective: str | None = None) -> list[Score]:
        return _score_for(self, message.get_piece(), None)


class TestCapabilities:
    def test_message_scorer_reports_validator_evidence_and_condition(self) -> None:
        validator = ScorerPromptValidator(
            supported_data_types=["text", "image_path"],
            supported_roles=["assistant"],
            required_metadata=["source"],
        )
        capabilities = _TypedMessageScorer(validator=validator).capabilities

        assert capabilities == ScorerCapabilities(
            condition_types=frozenset({_TypedCondition}),
            scorable_types=frozenset({MessageScorable, ContentScorable, ContentEntryScorable}),
            supported_data_types=frozenset({"text", "image_path"}),
            supported_roles=frozenset({"assistant"}),
            required_metadata=frozenset({"source"}),
        )

    def test_capabilities_are_frozen(self) -> None:
        capabilities = _TypedMessageScorer().capabilities
        with pytest.raises(dataclasses.FrozenInstanceError):
            capabilities.condition_types = frozenset()  # type: ignore[misc]

    def test_validator_defaults_exclude_simulated_roles(self) -> None:
        capabilities = _TypedMessageScorer().capabilities

        assert capabilities.supported_roles is not None
        assert "assistant" in capabilities.supported_roles
        assert "simulated_assistant" not in capabilities.supported_roles
        assert capabilities.required_metadata == frozenset()

    async def test_class_condition_takes_precedence_over_validator_flag_async(self) -> None:
        scorer = _TypedMessageScorer(validator=ScorerPromptValidator(is_objective_required=True))

        assert scorer.capabilities.condition_types == frozenset({_TypedCondition})
        expectation = ScoringExpectation(conditions=(_TypedCondition(),))
        [score] = await scorer.score_async(scorable=ContentScorable(value="hi"), expectation=expectation)
        assert score.get_value() is True
        assert score.scored_expectation == expectation

    @pytest.mark.parametrize(("required", "expected"), [(True, MatchesObjective), (False, None)])
    def test_undeclared_scorer_infers_condition_from_validator(
        self, required: bool, expected: type[Condition] | None
    ) -> None:
        scorer = _UndeclaredMessageScorer(validator=ScorerPromptValidator(is_objective_required=required))

        assert scorer.capabilities.condition_types == (frozenset({expected}) if expected else frozenset())

    def test_scorer_without_piece_filter_still_reports_message_support(self) -> None:
        capabilities = ManualScorer(value=True, rationale="checked", user_identifier="reviewer").capabilities

        assert capabilities == ScorerCapabilities(scorable_types=frozenset({MessageScorable}))
        assert capabilities.supported_data_types is None
        assert capabilities.supported_roles is None

    def test_wrapper_reports_derived_coverage_without_its_own_piece_filter(self) -> None:
        wrapper = TrueFalseInverterScorer(scorer=_TypedMessageScorer())

        assert wrapper.condition_type is None
        assert wrapper.capabilities.condition_types == frozenset({_TypedCondition})
        assert wrapper.capabilities.scorable_types == _TypedMessageScorer().capabilities.scorable_types
        assert wrapper.capabilities.supported_data_types is None
        assert wrapper.get_condition_types() == frozenset({_TypedCondition})

    def test_mixed_tree_reports_common_scorable_types_and_combined_conditions(self) -> None:
        wrapper = TrueFalseInverterScorer(
            scorer=TrueFalseCompositeScorer(
                scorers=[_TypedMessageScorer(), MessageToolCallScorer()],
                aggregator=TrueFalseScoreAggregator.AND,
            )
        )
        assert wrapper.capabilities.condition_types == frozenset({_TypedCondition, ToolsCalled})
        assert wrapper.capabilities.scorable_types == frozenset({MessageScorable})
        assert wrapper.capabilities.supported_roles is None

    def test_conversation_capabilities_do_not_filter_trigger_types_or_metadata(self) -> None:
        scorer = create_conversation_scorer(
            scorer=_TypedMessageScorer(),
            validator=ScorerPromptValidator(
                supported_data_types=["text"], required_metadata=["unused"], supported_roles=["assistant"]
            ),
        )
        assert scorer.capabilities.scorable_types == scorer.get_supported_scorable_types()
        assert MessageScorable in scorer.capabilities.scorable_types
        assert scorer.capabilities.supported_data_types is None
        assert scorer.capabilities.required_metadata == frozenset()
        assert scorer.capabilities.supported_roles == frozenset({"assistant"})

    def test_adapted_legacy_scorer_reports_its_message_evidence(self) -> None:
        with pytest.warns(DeprecationWarning, match="Scorer.__init__"):
            scorer = _LegacyDirectScorer()
        assert scorer.capabilities.scorable_types == _TypedMessageScorer().capabilities.scorable_types
        assert scorer.capabilities.supported_data_types == frozenset({"audio_path"})
        VideoHelper._validate_audio_scorer(scorer)

    def test_program_scorer_reports_code_and_execution_evidence(self) -> None:
        from pyrit.score.observation.observation_source import ObservationSource

        scorer = ProgramExecutionScorer(source=MagicMock(spec=ObservationSource))
        assert scorer.capabilities.condition_types == frozenset({ProgramBehaves})
        assert scorer.capabilities.scorable_types == frozenset(
            {MessageScorable, ContentScorable, ContentEntryScorable, ProgramExecutionScorable}
        )
        assert scorer.capabilities.supported_data_types is None

    def test_capabilities_cannot_be_overridden(self) -> None:
        with pytest.raises(TypeError, match="cannot override derived condition capabilities"):
            type("InvalidScorer", (_TypedMessageScorer,), {"capabilities": ScorerCapabilities()})

    def test_evidence_hook_cannot_replace_derived_coverage(self) -> None:
        class CustomEvidenceScorer(_GenericScorer):
            CONDITION_TYPE = _TypedCondition

            def _get_evidence_capabilities(self) -> ScorerCapabilities:
                return ScorerCapabilities(
                    condition_types=frozenset({MatchesObjective}),
                    scorable_types=frozenset({MessageScorable}),
                    required_metadata=frozenset({"trace"}),
                )

        capabilities = CustomEvidenceScorer().capabilities

        assert capabilities.condition_types == frozenset({_TypedCondition})
        assert capabilities.scorable_types == frozenset()
        assert capabilities.required_metadata == frozenset({"trace"})


class TestObjectiveScorersDeclareMatchesObjective:
    @pytest.mark.parametrize("scorer_type", [SelfAskScaleScorer, DecisionScorer])
    def test_declares_matches_objective_without_validator_flag(self, scorer_type: type) -> None:
        assert scorer_type.CONDITION_TYPE is MatchesObjective
        assert scorer_type._DEFAULT_VALIDATOR.is_objective_required is False

    def test_custom_validator_cannot_drop_objective_requirement(self) -> None:
        scorer = SelfAskScaleScorer(
            chat_target=MagicMock(),
            system_prompt="Rate the response.",
            scale=MagicMock(),
            validator=ScorerPromptValidator(supported_data_types=["text", "image_path"]),
        )

        assert scorer.capabilities.condition_types == frozenset({MatchesObjective})
        with pytest.raises(ValueError, match="MatchesObjective requires the expectation to carry an objective"):
            scorer._validate_expectation(expectation=None)

    @pytest.mark.parametrize("configured_rubric", [False, True])
    def test_general_float_scorer_keeps_configurable_objective_requirement(self, configured_rubric: bool) -> None:
        scorer = SelfAskGeneralFloatScaleScorer(
            chat_target=MagicMock(),
            system_prompt_format_string="Rate this response.",
            scale=MagicMock(),
            validator=ScorerPromptValidator(supported_data_types=["text"]) if configured_rubric else None,
        )
        if configured_rubric:
            assert scorer.capabilities.condition_types == frozenset()
            scorer.prepare_expectation(expectation=None)
        else:
            assert scorer.capabilities.condition_types == frozenset({MatchesObjective})
            with pytest.raises(ValueError, match="MatchesObjective requires"):
                scorer.prepare_expectation(expectation=None)


class TestNoAmbientExpectation:
    def test_ambient_expectation_context_is_removed(self) -> None:
        removed = ("_CURRENT_SCORING_EXPECTATION", "_scoring_expectation_context", "_get_current_scoring_expectation")
        for name in removed:
            assert not hasattr(execution, name)


class TestLegacyHookConstruction:
    def test_error_is_a_type_error(self) -> None:
        assert issubclass(LegacyScorerHookError, TypeError)

    def test_typed_condition_with_only_legacy_piece_hook_fails_at_construction(self) -> None:
        class TypedLegacyScorer(_LegacyPieceScorer):
            CONDITION_TYPE = _TypedCondition

        with pytest.raises(
            LegacyScorerHookError,
            match=r"TypedLegacyScorer declares _TypedCondition.*Implement _score_piece_with_expectation_async",
        ):
            TypedLegacyScorer()

    def test_typed_condition_with_legacy_message_hook_fails_at_construction(self) -> None:
        class TypedMessageHookScorer(_TypedMessageScorer):
            async def _score_async(self, message: Message, *, objective: str | None = None) -> list[Score]:
                raise AssertionError("must not run")

        with pytest.raises(LegacyScorerHookError, match=r"Implement _score_async\(\.\.\., expectation=\.\.\.\)"):
            TypedMessageHookScorer()

    @pytest.mark.parametrize("condition_type", [None, MatchesObjective])
    async def test_objective_based_legacy_hook_still_scores_async(self, condition_type: type[Condition] | None) -> None:
        legacy_type = type("ObjectiveLegacyScorer", (_LegacyPieceScorer,), {"CONDITION_TYPE": condition_type})
        expectation = ScoringExpectation(objective="say hi") if condition_type else None

        [score] = await legacy_type().score_async(scorable=ContentScorable(value="hi"), expectation=expectation)

        assert score.get_value() is True

    def test_failure_does_not_affect_sibling_classes(self) -> None:
        class TypedLegacyScorer(_LegacyPieceScorer):
            CONDITION_TYPE = _TypedCondition

        class ObjectiveLegacyScorer(_LegacyPieceScorer):
            CONDITION_TYPE = MatchesObjective

        with pytest.raises(LegacyScorerHookError):
            TypedLegacyScorer()
        assert ObjectiveLegacyScorer().condition_type is MatchesObjective
        assert _TypedMessageScorer().condition_type is _TypedCondition

    def test_direct_scorer_with_typed_legacy_hook_fails_at_construction(self) -> None:
        class TypedDirectScorer(TrueFalseScorer):
            CONDITION_TYPE = _TypedCondition

            def _build_identifier(self) -> ComponentIdentifier:
                return self._create_identifier()

            async def _score_async(self, message: Message, *, objective: str | None = None) -> list[Score]:
                raise AssertionError("must not run")

        with pytest.raises(LegacyScorerHookError, match="Implement _score_scorable_async"):
            TypedDirectScorer()

    def test_subclass_of_adapted_legacy_scorer_cannot_add_typed_condition(self) -> None:
        typed_child = type("TypedChild", (_LegacyDirectScorer,), {"CONDITION_TYPE": _TypedCondition})
        with pytest.raises(LegacyScorerHookError, match="TypedChild declares _TypedCondition"):
            typed_child()

    async def test_custom_pipeline_delegation_keeps_runtime_guard_async(self) -> None:
        class CustomPipelineScorer(_LegacyPieceScorer):
            CONDITION_TYPE = _TypedCondition

            async def _score_prepared_message_async(
                self, *, message: Message, expectation: ScoringExpectation | None
            ) -> list[Score]:
                return await super()._score_prepared_message_async(message=message, expectation=expectation)

        scorer = CustomPipelineScorer()
        with pytest.raises(LegacyScorerHookError, match="matched typed conditions"):
            await scorer.score_async(
                scorable=ContentScorable(value="hi"),
                expectation=ScoringExpectation(conditions=(_TypedCondition(),)),
            )

    async def test_custom_pipeline_can_evaluate_typed_condition_without_piece_hook_async(self) -> None:
        class CustomPipelineScorer(_LegacyPieceScorer):
            CONDITION_TYPE = _TypedCondition

            async def _score_prepared_message_async(
                self, *, message: Message, expectation: ScoringExpectation | None
            ) -> list[Score]:
                self._get_required_condition(expectation=expectation, condition_type=_TypedCondition)
                return _score_for(self, message.get_piece(), expectation)

        [score] = await CustomPipelineScorer().score_async(
            scorable=ContentScorable(value="hi"),
            expectation=ScoringExpectation(conditions=(_TypedCondition(),)),
        )
        assert score.get_value() is True


class TestModalityPreflight:
    def test_video_audio_scorer_without_message_evidence_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"supports 'audio_path' data type\. Supported types: none reported"):
            VideoHelper._validate_audio_scorer(_GenericScorer())

    def test_video_audio_scorer_without_audio_is_rejected(self) -> None:
        scorer = _TypedMessageScorer(validator=ScorerPromptValidator(supported_data_types=["text"]))
        with pytest.raises(ValueError, match=r"Supported types: \['text'\]"):
            VideoHelper._validate_audio_scorer(scorer)

    def test_video_audio_scorer_with_audio_is_accepted(self) -> None:
        scorer = _TypedMessageScorer(validator=ScorerPromptValidator(supported_data_types=["audio_path"]))
        VideoHelper._validate_audio_scorer(scorer)

    def test_audio_text_scorer_without_text_is_rejected(self) -> None:
        scorer = _TypedMessageScorer(validator=ScorerPromptValidator(supported_data_types=["image_path"]))
        with pytest.raises(ValueError, match=r"supports 'text' data type\. Supported types: \['image_path'\]"):
            AudioTranscriptHelper._validate_text_scorer(scorer)
