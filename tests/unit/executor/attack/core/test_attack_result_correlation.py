# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import json

import pytest
from unit.mocks import MockPromptTarget

from pyrit.executor.attack import (
    AttackAdversarialConfig,
    AttackParameters,
    AttackScoringConfig,
    PromptSendingAttack,
    RedTeamingAttack,
    SingleTurnAttackContext,
)
from pyrit.memory import SQLiteMemory
from pyrit.models import AttackOutcome, Message, MessagePiece
from pyrit.score import SelfAskRefusalScorer, SubStringScorer

KEY = MessagePiece.ATTACK_RESULT_ID_METADATA_KEY

pytestmark = pytest.mark.usefixtures("patch_central_database")


class _RecordingTarget(MockPromptTarget):
    """Record the attack result ID each request carries when it reaches the target."""

    def __init__(self, *, reply: str = "default", fail: bool = False) -> None:
        super().__init__()
        self.received_ids: list[str | None] = []
        self._reply = reply
        self._fail = fail

    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        request = normalized_conversation[-1].get_piece()
        self.received_ids.append(request.prompt_metadata.get(KEY))
        if self._fail:
            raise RuntimeError("target unavailable")
        return [
            MessagePiece(
                role="assistant", original_value=self._reply, conversation_id=request.conversation_id
            ).to_message()
        ]


async def _pieces_async(*, memory: SQLiteMemory, conversation_ids: set[str]) -> list[MessagePiece]:
    return [
        piece
        for conversation_id in conversation_ids
        for piece in await memory.get_message_pieces_async(conversation_id=conversation_id)
    ]


async def test_each_execution_allocates_one_result_id_async(sqlite_instance: SQLiteMemory) -> None:
    target = _RecordingTarget()
    attack = PromptSendingAttack(objective_target=target)
    context = SingleTurnAttackContext(params=AttackParameters(objective="first"))
    assert context.attack_result_id is None

    first = await attack.execute_with_context_async(context=context)
    first_id = context.attack_result_id
    second = await attack.execute_with_context_async(context=context)

    assert first_id == first.attack_result_id
    assert context.attack_result_id == second.attack_result_id
    assert first.attack_result_id != second.attack_result_id
    assert target.received_ids == [first.attack_result_id, second.attack_result_id]
    stored = await sqlite_instance.get_attack_results_async(
        attack_result_ids=[first.attack_result_id, second.attack_result_id]
    )
    assert {result.attack_result_id for result in stored} == {first.attack_result_id, second.attack_result_id}


async def test_concurrent_executions_keep_their_own_result_id_async(sqlite_instance: SQLiteMemory) -> None:
    target = _RecordingTarget()
    attack = PromptSendingAttack(objective_target=target)

    results = await asyncio.gather(*(attack.execute_async(objective=f"objective {i}") for i in range(3)))

    assert len({result.attack_result_id for result in results}) == 3
    for result in results:
        [request] = [
            piece
            for piece in await sqlite_instance.get_message_pieces_async(conversation_id=result.conversation_id)
            if piece.role == "user"
        ]
        assert request.prompt_metadata[KEY] == result.attack_result_id


async def test_result_id_reaches_every_conversation_of_a_multi_turn_attack_async(sqlite_instance: SQLiteMemory) -> None:
    objective_target = _RecordingTarget()
    adversarial_chat = _RecordingTarget(
        reply=json.dumps({"next_message": "next", "rationale": "r", "last_response_summary": "s"})
    )
    attack = RedTeamingAttack(
        objective_target=objective_target,
        attack_adversarial_config=AttackAdversarialConfig(target=adversarial_chat),
        attack_scoring_config=AttackScoringConfig(objective_scorer=SubStringScorer(substring="never present")),
        max_turns=2,
    )

    result = await attack.execute_async(objective="objective")

    assert result.outcome == AttackOutcome.FAILURE
    conversation_ids = result.get_all_conversation_ids()
    assert len(conversation_ids) == 2
    pieces = await _pieces_async(memory=sqlite_instance, conversation_ids=conversation_ids)
    requests = [piece for piece in pieces if piece.role == "user"]
    assert {piece.conversation_id for piece in requests} == conversation_ids
    assert all(piece.prompt_metadata[KEY] == result.attack_result_id for piece in requests)
    assert all(KEY not in piece.prompt_metadata for piece in pieces if piece.role != "user")
    assert set(objective_target.received_ids) == {result.attack_result_id}
    assert set(adversarial_chat.received_ids) == {result.attack_result_id}

    correlated = await sqlite_instance.get_message_pieces_async(prompt_metadata={KEY: result.attack_result_id})
    assert {piece.id for piece in correlated} == {piece.id for piece in requests}
    [stored] = await sqlite_instance.get_attack_results_async(attack_result_ids=[result.attack_result_id])
    assert stored.get_all_conversation_ids() == conversation_ids


async def test_scoring_requests_during_the_attack_are_correlated_async(sqlite_instance: SQLiteMemory) -> None:
    verdict = json.dumps({"score_value": "false", "rationale": "r", "description": "d", "metadata": ""})
    judge = _RecordingTarget(reply=verdict)
    attack = PromptSendingAttack(
        objective_target=_RecordingTarget(),
        attack_scoring_config=AttackScoringConfig(objective_scorer=SelfAskRefusalScorer(chat_target=judge)),
    )

    result = await attack.execute_async(objective="objective")

    assert result.automated_score is not None
    assert judge.received_ids == [result.attack_result_id]
    correlated = await sqlite_instance.get_message_pieces_async(prompt_metadata={KEY: result.attack_result_id})
    conversation_ids = {piece.conversation_id for piece in correlated}
    assert len(conversation_ids) == 2
    assert result.conversation_id in conversation_ids


async def test_error_result_keeps_the_allocated_result_id_async(sqlite_instance: SQLiteMemory) -> None:
    target = _RecordingTarget(fail=True)
    attack = PromptSendingAttack(objective_target=target)
    context = SingleTurnAttackContext(params=AttackParameters(objective="objective"))

    with pytest.raises(RuntimeError):
        await attack.execute_with_context_async(context=context)

    assert context.attack_result_id is not None
    assert target.received_ids == [context.attack_result_id]
    [stored] = await sqlite_instance.get_attack_results_async(attack_result_ids=[context.attack_result_id])
    assert stored.outcome == AttackOutcome.ERROR
    correlated = await sqlite_instance.get_message_pieces_async(prompt_metadata={KEY: context.attack_result_id})
    assert [piece.role for piece in correlated] == ["user"]


async def test_copied_history_does_not_carry_another_result_id_async(sqlite_instance: SQLiteMemory) -> None:
    target = _RecordingTarget()
    attack = PromptSendingAttack(objective_target=target)
    first = await attack.execute_async(objective="first")
    history = list(await sqlite_instance.get_conversation_messages_async(conversation_id=first.conversation_id))
    assert history[0].get_piece().prompt_metadata[KEY] == first.attack_result_id

    second = await attack.execute_async(objective="second", prepended_conversation=history)

    pieces = await sqlite_instance.get_message_pieces_async(conversation_id=second.conversation_id)
    copied = [piece for piece in pieces if piece.prompt_metadata.get(MessagePiece.PREPENDED_HISTORY_METADATA_KEY)]
    assert len(copied) == 2
    assert all(KEY not in piece.prompt_metadata for piece in copied)
    live = [piece for piece in pieces if piece.role == "user" and piece not in copied]
    assert [piece.prompt_metadata[KEY] for piece in live] == [second.attack_result_id]
    assert history[0].get_piece().prompt_metadata[KEY] == first.attack_result_id
    correlated = await sqlite_instance.get_message_pieces_async(prompt_metadata={KEY: first.attack_result_id})
    assert {piece.conversation_id for piece in correlated} == {first.conversation_id}
