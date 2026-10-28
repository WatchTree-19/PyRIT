# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Declare what a scorer accepts, so a mismatch is reported before evidence is acquired."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrit.models import ChatMessageRole, Condition, PromptDataType, Scorable
    from pyrit.score.scorer_prompt_validator import ScorerPromptValidator


class LegacyScorerHookError(TypeError):
    """
    A scorer's protected hook cannot receive the typed criterion the scorer declares.

    Legacy hooks receive only the objective. They keep working for objective-based and
    constructor-configured scoring until 2.0, but a scorer that declares a typed condition
    must implement an expectation-aware hook so the criterion reaches the code that matches it.
    """


@dataclass(frozen=True, kw_only=True)
class ScorerCapabilities:
    """
    The evidence and criteria a scorer accepts.

    Capabilities decide whether a scorer applies, never how an applicable input fails: piece
    limits and error policy stay on the message validator. Condition and scorable coverage are
    derived from the whole scorer tree. The message fields describe only the filter that this
    scorer applies to the pieces of the message it scores. A wrapper applies no filter of its
    own, because each child filters its own evidence.

    Attributes:
        condition_types (frozenset[type[Condition]]): The conditions the scorer tree covers.
        scorable_types (frozenset[type[Scorable]]): The evidence references the scorer accepts.
        supported_data_types (frozenset[PromptDataType] | None): Message piece data types the
            scorer reads, or None when it applies no data-type filter of its own.
        supported_roles (frozenset[ChatMessageRole] | None): Message roles the scorer reads, or
            None when it applies no role filter of its own.
        required_metadata (frozenset[str]): Metadata keys a scored message piece must carry.
    """

    condition_types: frozenset[type[Condition]] = frozenset()
    scorable_types: frozenset[type[Scorable]] = frozenset()
    supported_data_types: frozenset[PromptDataType] | None = None
    supported_roles: frozenset[ChatMessageRole] | None = None
    required_metadata: frozenset[str] = frozenset()

    @classmethod
    def from_validator(cls, validator: ScorerPromptValidator) -> ScorerCapabilities:
        """
        Describe the piece filter of a message validator.

        Args:
            validator (ScorerPromptValidator): The validator that filters the scored pieces.

        Returns:
            ScorerCapabilities: The validator's data types, roles, and required metadata.
        """
        return cls(
            supported_data_types=frozenset(validator.supported_data_types),
            supported_roles=frozenset(validator.supported_roles),
            required_metadata=frozenset(validator.required_metadata),
        )
