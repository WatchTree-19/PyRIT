# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrit.prompt_target import PromptTarget

from pyrit.models import (
    ComponentIdentifier,
    Condition,
    Scorable,
    Score,
    ScoringExpectation,
)
from pyrit.score.float_scale.float_scale_scorer import FloatScaleScorer
from pyrit.score.observation import _merge_observation_ids

#: Metadata key naming which scorer produced the returned value: ``"primary"`` or ``"fallback"``.
RESOLVED_BY_KEY = "resolved_by"
#: Metadata key carrying the primary scorer's rationale when the fallback produced the value.
PRIMARY_RATIONALE_KEY = "primary_rationale"


class FloatScaleFallbackScorer(FloatScaleScorer):
    """
    Route a primary scorer's abstentions to a fallback scorer.

    Some float-scale scorers decline to call a response and return
    ``ScoreStatus.UNDETERMINED``, for instance a classifier whose calibrated probability
    falls inside an abstain band. That abstention is only useful if a caller does
    something with it. This scorer is that caller: it scores with ``scorer`` first, and
    when any of the returned scores is undetermined it scores again with
    ``fallback_scorer`` and returns that result instead.

    The typical pairing is a cheap, fast primary that covers the bulk of responses and an
    LLM judge as the fallback that only runs on the uncertain tail, so the judge's cost is
    paid only where the primary could not decide.

    Every returned score records which scorer produced it in
    ``score_metadata[RESOLVED_BY_KEY]``. When the fallback produced it, the primary's
    rationale is kept in ``score_metadata[PRIMARY_RATIONALE_KEY]`` and the primary's own
    metadata (an abstain band, a calibrated probability) is merged in, so nothing the
    primary said is lost. If the fallback abstains as well, the returned score is
    undetermined and says so.
    """

    def __init__(self, *, scorer: FloatScaleScorer, fallback_scorer: FloatScaleScorer) -> None:
        """
        Initialize the FloatScaleFallbackScorer.

        Args:
            scorer (FloatScaleScorer): The primary scorer, tried first on every scorable.
            fallback_scorer (FloatScaleScorer): The scorer consulted only when the primary
                returns an undetermined score.

        Raises:
            ValueError: If either scorer is not a float scale scorer, or if the same object is
                passed as both.
        """
        if not isinstance(scorer, FloatScaleScorer):
            raise ValueError("The scorer must be a float scale scorer")
        if not isinstance(fallback_scorer, FloatScaleScorer):
            raise ValueError("The fallback scorer must be a float scale scorer")
        if scorer is fallback_scorer:
            raise ValueError("The fallback scorer must be a different scorer from the primary")
        self._scorer = scorer
        self._fallback_scorer = fallback_scorer

        super().__init__()

    def _build_identifier(self) -> ComponentIdentifier:
        """
        Build the identifier for this scorer.

        Returns:
            ComponentIdentifier: The identifier for this scorer.
        """
        return self._create_identifier(
            sub_scorers=[self._scorer.get_identifier(), self._fallback_scorer.get_identifier()],
        )

    def get_chat_target(self) -> "PromptTarget | None":
        """
        Return the primary scorer's chat target, or the fallback's when the primary has none.

        Returns:
            PromptTarget | None: The chat target, if either wrapped scorer has one.
        """
        target = self._scorer.get_chat_target()
        if target is not None:
            return target
        return self._fallback_scorer.get_chat_target()

    def matched_conditions(self) -> frozenset[type[Condition]]:
        """
        Report what either wrapped scorer matches.

        Returns:
            frozenset[type[Condition]]: The union of the condition types the wrapped scorers route.
        """
        return self._scorer.matched_conditions() | self._fallback_scorer.matched_conditions()

    def required_conditions(self) -> frozenset[type[Condition]]:
        """
        Report what both wrapped scorers require.

        Returns:
            frozenset[type[Condition]]: The union of the required condition types.
        """
        return self._scorer.required_conditions() | self._fallback_scorer.required_conditions()

    def _validate_expectation(self, *, expectation: ScoringExpectation | None) -> None:
        """Validate wrapper and both child criteria without checking sibling condition coverage."""
        super()._validate_expectation(expectation=expectation)
        self._scorer._validate_expectation(expectation=expectation)
        self._fallback_scorer._validate_expectation(expectation=expectation)

    async def _score_scorable_async(
        self,
        *,
        scorable: Scorable,
        expectation: ScoringExpectation | None,
    ) -> list[Score]:
        """
        Score with the primary scorer, and with the fallback if the primary abstained.

        Args:
            scorable (Scorable): What to look at.
            expectation (ScoringExpectation | None): What the wrapped scorers should look for.

        Returns:
            list[Score]: ``[]`` when the primary scorer is non-applicable; the primary's scores
                when all of them are complete; otherwise the fallback scorer's scores, each
                carrying the primary's rationale and metadata.
        """
        primary_scores = await self._scorer._score_nested_async(scorable=scorable, expectation=expectation)
        if not primary_scores:
            return []

        if not any(score.is_undetermined for score in primary_scores):
            return [self._stamp(score, resolved_by="primary") for score in primary_scores]

        fallback_scores = await self._fallback_scorer._score_nested_async(scorable=scorable, expectation=expectation)
        if not fallback_scores:
            # The fallback had nothing to say about this scorable at all, so the primary's
            # abstention stands as the answer.
            return [self._stamp(score, resolved_by="primary") for score in primary_scores]

        return [self._stamp(score, resolved_by="fallback", primary_scores=primary_scores) for score in fallback_scores]

    def _stamp(
        self,
        score: Score,
        *,
        resolved_by: str,
        primary_scores: list[Score] | None = None,
    ) -> Score:
        """
        Re-issue a wrapped score as this scorer's own, recording which scorer produced it.

        Returns:
            Score: The same score object, re-identified and annotated.
        """
        metadata: dict[str, str | int | float] = dict(score.score_metadata or {})
        metadata[RESOLVED_BY_KEY] = resolved_by

        if primary_scores is not None:
            primary = primary_scores[0]
            primary_type = self._scorer.get_identifier().class_name
            fallback_type = self._fallback_scorer.get_identifier().class_name
            # Keep everything the primary said. Its own metadata (an abstain band, a
            # calibrated probability) goes in first so the fallback's keys win on a clash.
            metadata = {**(primary.score_metadata or {}), **metadata}
            if primary.score_rationale:
                metadata[PRIMARY_RATIONALE_KEY] = primary.score_rationale
            verdict = "also abstained" if score.is_undetermined else f"returned {score.score_value}"
            score.score_rationale = (
                f"{primary_type} abstained, so the score was routed to {fallback_type}, which {verdict}.\n"
                f"{score.score_rationale or ''}"
            ).rstrip()
            score.observation_ids = _merge_observation_ids(scores=[*primary_scores, score])

        score.score_metadata = metadata
        score.id = uuid.uuid4()
        score.scorer_class_identifier = self.get_identifier()
        return score
