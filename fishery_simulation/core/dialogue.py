"""
Genuine multi-turn dyadic conversation between two agents.
One LLM call per turn. Both agents read and respond to each other.

Past conversations with the same partner are retrieved from memory and
shown explicitly — the memory stream is the only persistent state.
Harvest history is passed in explicitly from the environment.
"""
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.environment import FisheryEnvironment as _FE

if TYPE_CHECKING:
    from core.memory import MemoryStream
    from core.llm import LLMAdapter

SPEECH_ACTS = {
    "proposal", "warning", "agreement", "disagreement",
    "question", "statement", "closing"
}

IMPORTANCE_BY_ACT = {
    "agreement": 9,
    "proposal": 8,
    "warning": 8,
    "disagreement": 7,
    "statement": 4,
    "question": 4,
    "closing": 3,
}

# Minimum meaningful utterance length (chars)
_MIN_UTTERANCE_LEN = 8


def _clean(text: str) -> str:
    """Strip leading/trailing whitespace and blank lines."""
    text = re.sub(r"\n{2,}", "\n", text.strip())
    return text.strip()


@dataclass
class DialogueTurn:
    speaker_id: str
    listener_id: str
    content: str
    tick: int
    turn_number: int
    speech_act: str  # one of SPEECH_ACTS


class DialogueEngine:
    def __init__(self, llm):
        self.llm = llm

    def run(
        self,
        initiator_id: str,
        responder_id: str,
        initiator_memory,
        responder_memory,
        initiator_persona: str,
        responder_persona: str,
        opening_message: str,
        world_nl: str,
        tick: int,
        max_turns: int = 6,
        harvest_history_nl: str = "",
    ) -> list:
        """
        Full conversation loop.
        Turn 0 : initiator's opening_message (already formed from CoT).
        Turn 1+ : alternating LLM calls, each with full memory + harvest history.
        All turns stored in both agents' memories immediately.
        """
        history = []

        # Turn 0: initiator's opening
        speech_act_0  = self._classify_speech_act(opening_message)
        opening_clean = _clean(opening_message)
        turn0 = DialogueTurn(
            speaker_id=initiator_id,
            listener_id=responder_id,
            content=opening_clean,
            tick=tick,
            turn_number=0,
            speech_act=speech_act_0,
        )
        history.append(turn0)
        self._store_turn(turn0, initiator_memory, "sent")
        self._store_turn(turn0, responder_memory, "received")

        # Alternating turns
        for turn_num in range(1, max_turns):
            if turn_num % 2 == 1:
                speaker_id      = responder_id
                listener_id     = initiator_id
                speaker_persona = responder_persona
                speaker_memory  = responder_memory
                listener_memory = initiator_memory
            else:
                speaker_id      = initiator_id
                listener_id     = responder_id
                speaker_persona = initiator_persona
                speaker_memory  = initiator_memory
                listener_memory = responder_memory

            utterance, speech_act = self._generate_turn(
                speaker_id=speaker_id,
                listener_id=listener_id,
                speaker_persona=speaker_persona,
                speaker_memory=speaker_memory,
                history=history,
                world_nl=world_nl,
                tick=tick,
                harvest_history_nl=harvest_history_nl,
            )

            turn = DialogueTurn(
                speaker_id=speaker_id,
                listener_id=listener_id,
                content=utterance,
                tick=tick,
                turn_number=turn_num,
                speech_act=speech_act,
            )
            history.append(turn)

            # Store in both memories immediately
            self._store_turn(turn, speaker_memory, "sent")
            self._store_turn(turn, listener_memory, "received")

            if speech_act == "closing":
                break

        return history

    # ── Internal helpers ──────────────────────────────────────────

    def _retrieve_context_for(
        self,
        speaker_memory,
        listener_id: str,
        tick: int,
    ) -> str:
        """
        Past conversations with this specific partner (top 5) plus
        recent norm beliefs, reflections, decisions (top 3).
        """
        past_dialogue = speaker_memory.retrieve(
            query=listener_id,
            top_k=5,
            current_tick=tick,
            type_filter=["dialogue"],
        )
        general = speaker_memory.retrieve(
            query=f"stock harvest norm {listener_id}",
            top_k=3,
            current_tick=tick,
            type_filter=["norm_belief", "reflection", "harvest_decision"],
        )

        # Sort dialogue turns chronologically (oldest first) so the
        # conversation reads top-to-bottom in the order it happened.
        past_dialogue_sorted = sorted(past_dialogue, key=lambda m: m.created_tick)

        seen, combined = set(), []
        for m in past_dialogue_sorted + general:
            if id(m) not in seen:
                seen.add(id(m))
                combined.append(m)

        return speaker_memory.format_for_prompt(combined)

    def _generate_turn(
        self,
        speaker_id: str,
        listener_id: str,
        speaker_persona: str,
        speaker_memory,
        history: list,
        world_nl: str,
        tick: int,
        harvest_history_nl: str = "",
    ) -> tuple:
        """One LLM call — generates next utterance."""
        memory_text = self._retrieve_context_for(speaker_memory, listener_id, tick)

        # Skip empty-content turns so the LLM doesn't see blank speaker lines
        history_lines = [
            f"{t.speaker_id}: {t.content}"
            for t in history
            if t.content.strip()
        ]
        history_text = (
            "\n".join(history_lines)
            if history_lines
            else "(conversation just started)"
        )

        harvest_section = (
            f"\nHarvest history:\n{harvest_history_nl}\n"
            if harvest_history_nl else ""
        )

        prompt = (
            f"You are {speaker_id}, a fisher in conversation with {listener_id}.\n\n"
            f"{speaker_persona}\n\n"
            f"Situation: {world_nl}"
            f"{harvest_section}\n"
            f"What you remember about {listener_id} and past conversations:\n"
            f"{memory_text}\n\n"
            f"Conversation so far:\n"
            f"{history_text}\n\n"
            f"Respond as {speaker_id} in 1 to 3 plain sentences. "
            f"No markdown. No quotation marks wrapping your reply. "
            f"Be direct. If the conversation is naturally finished, wrap it up.\n\n"
            f"After your reply, on a new line write:\n"
            f"SPEECH_ACT: [proposal/warning/agreement/disagreement/question/statement/closing]"
        )

        raw = self.llm.complete(
            prompt, max_tokens=2000,
            label=f"dialogue {speaker_id}→{listener_id} tick={tick}",
        )
        utterance, speech_act = self._parse_turn(raw)
        return utterance, speech_act

    def _parse_turn(self, raw: str) -> tuple:
        """
        Extract (utterance, speech_act) from raw LLM output.
        Handles missing / misplaced SPEECH_ACT label and markdown noise.
        """
        if not raw or not raw.strip():
            return "", ""

        speech_act = "statement"
        raw_stripped = raw.strip()

        if "SPEECH_ACT:" in raw_stripped:
            parts      = raw_stripped.split("SPEECH_ACT:", 1)
            pre        = parts[0].strip()
            act_raw    = parts[1].strip().lower()
            first_word = re.sub(r"[^a-z]", "", act_raw.split()[0]) if act_raw.split() else ""
            if first_word in SPEECH_ACTS:
                speech_act = first_word

            if pre:
                utterance = _clean(pre)
            else:
                # SPEECH_ACT appeared first — rest of string after the act word is the utterance
                rest = re.sub(rf"^{re.escape(first_word)}", "", parts[1].strip(),
                              count=1, flags=re.IGNORECASE).strip()
                utterance = _clean(rest) if rest else _clean(raw_stripped)
        else:
            utterance = _clean(raw_stripped)

        # Strip wrapping quotes and markdown bold/italic only if they wrap the whole string
        utterance = re.sub(r'^[\*_]{1,2}|[\*_]{1,2}$', '', utterance).strip()
        utterance = re.sub(r'^["\']|["\']$', '', utterance).strip()

        # Fallback if still too short
        if len(utterance) < _MIN_UTTERANCE_LEN:
            # Take everything before SPEECH_ACT line, cleaned
            candidate = _clean(re.sub(r"SPEECH_ACT:.*$", "", raw_stripped, flags=re.DOTALL))
            utterance = candidate if len(candidate) >= _MIN_UTTERANCE_LEN else raw_stripped

        # Strip any leftover SPEECH_ACT line that ended up in the utterance
        utterance = re.sub(r"SPEECH_ACT:\s*\w+", "", utterance).strip()

        if not utterance or len(utterance) < _MIN_UTTERANCE_LEN:
            utterance = ""

        return utterance, speech_act

    def _classify_speech_act(self, text: str) -> str:
        """Keyword classification for the opening message (turn 0)."""
        t = text.lower()
        if any(w in t for w in ["propose", "suggest", "should we", "what if we",
                                  "limit", "quota", "how about", "cap", "agree to"]):
            return "proposal"
        if any(w in t for w in ["too much", "took more", "noticed you",
                                  "concerned about", "worried about", "taking a lot"]):
            return "warning"
        if any(w in t for w in ["agree", "sounds good", "i'm in",
                                  "fair enough", "that works"]):
            return "agreement"
        if t.rstrip().endswith("?") or "what do you think" in t or "do you agree" in t:
            return "question"
        if any(w in t for w in ["goodbye", "that's all", "i'll leave it there", "see you"]):
            return "closing"
        return "statement"

    def _store_turn(self, turn: "DialogueTurn", memory, role: str):
        """
        Store a dialogue turn in memory.
        Content uses Year/Season label — NOT a raw tick prefix, because
        format_for_prompt() already prepends [tick N] from the MemoryObject.
        """
        importance   = IMPORTANCE_BY_ACT.get(turn.speech_act, 4)
        season_label = _FE.season_label(turn.tick)

        if role == "sent":
            content = (
                f"{season_label}: I said to {turn.listener_id}: "
                f"{turn.content!r} [{turn.speech_act}]"
            )
        else:
            content = (
                f"{season_label}: {turn.speaker_id} said to me: "
                f"{turn.content!r} [{turn.speech_act}]"
            )

        memory.add(
            content=content,
            importance=importance,
            memory_type="dialogue",
            tick=turn.tick,
        )
