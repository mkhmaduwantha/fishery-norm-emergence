"""
The core agent. Combines memory stream and LLM into a decision-making unit.
NO belief model class — all state in memory.
"""
import re
from core.memory import MemoryStream
from core.llm import LLMAdapter
from experiments.conditions import CONDITION_PROMPTS


PERSONAS = [
    # Economically comfortable, established
    "You are a fisher in your 50s who owns your boat outright. "
    "You have a stable income and fish because it is your livelihood "
    "and identity.",

    # Economically pressured, short horizon
    "You are a fisher in your 30s with a boat loan to pay off "
    "and two young children. This season's catch determines "
    "whether you can make your payments.",

    # New entrant, uncertain
    "You are new to this fishery, having moved here two years ago. "
    "You do not yet have the relationships or reputation "
    "that long-established fishers have.",

    # Highly dependent, no alternatives
    "You are a fisher with no other skills or income sources. "
    "Fishing is the only thing you know. A bad season "
    "means real hardship for your household.",

    # Experienced but struggling
    "You are a fisher in your 40s who has seen catches decline "
    "steadily over the past decade. You are not sure how much "
    "longer you can continue in this profession.",

    # Commercially oriented
    "You are a fisher who sells to a processor that rewards "
    "volume. Larger catches mean better prices and better "
    "relationships with your buyer.",

    # Part-time, lower dependence
    "You are a fisher who also has income from a land-based job. "
    "Fishing supplements your income but is not your only livelihood.",

    # Elder, long history
    "You are one of the oldest fishers in this community. "
    "You have fished here for 40 years and have seen many "
    "changes in the stock and in who fishes here.",
]


class FisheryAgent:
    def __init__(
        self,
        agent_id: str,
        persona: str,
        condition: str,
        llm: LLMAdapter,
    ):
        self.agent_id = agent_id
        self.persona = persona
        self.condition = condition
        self.llm = llm
        self.memory = MemoryStream()

    def decide(self, observation: dict, world_nl: str) -> dict:
        """Single LLM call producing harvest decision + communication intent."""
        tick = observation["tick"]

        # Build retrieval query
        others = list(observation.get("others_harvests", {}).keys())
        query = (
            f"stock {observation['stock_trend']} "
            f"{' '.join(others)} harvest season"
        )

        # Retrieve top 6 memories
        memories = self.memory.retrieve(query=query, top_k=6, current_tick=tick)
        memory_text = self.memory.format_for_prompt(memories)

        # Format others' harvests
        others_harvests = observation.get("others_harvests", {})
        my_last = observation.get("my_last_harvest")
        others_harvests_text = self._format_others_harvests(
            others_harvests, my_last, tick, observation
        )

        # Condition insert
        condition_insert = CONDITION_PROMPTS.get(self.condition, "")

        # Harvest history from environment (last 5 normal seasons + collapse context)
        harvest_history_text = observation.get("harvest_history_nl", "")

        prompt = f"""You are {self.agent_id}, a fisher who depends on a shared
fish stock for your livelihood.

{self.persona}

The fishing year has 4 seasons. The stock can collapse if overfished, after which
the fishery closes for the rest of that year and reopens the following year with
a recovered stock.

Current situation:
{world_nl}

Recent harvest history (last 5 seasons):
{harvest_history_text}

{condition_insert}

What you remember from past seasons:
{memory_text}

Reason through your decision step by step:

STOCK_ASSESSMENT: What is your reading of the stock right now?
OTHERS_BEHAVIOUR: What pattern do you notice in what others are taking? Is anyone taking significantly more or less?
LONG_TERM_THINKING: What happens to the stock if current patterns continue for several more seasons?
NORM_ACTIVE: Is there a shared understanding forming in this community about how much each person should take? Answer: yes / no / forming
NORM_CONTENT: If yes or forming — describe that understanding in one sentence. If no — write "none".
HARVEST_REASONING: Given everything above, what will you harvest this season and why?
HARVEST_AMOUNT: [a number between 0.0 and 10.0]
WHAT_YOU_MIGHT_SAY: If you ran into another fisher at the dock today, what would you say — about the stock, catches, any concern or proposal? Write "nothing" if you have nothing to say."""

        raw = self.llm.complete(
            prompt, max_tokens=2000,
            label=f"harvest {self.agent_id} tick={tick}",
        )
        parsed = self._parse_cot(raw, tick)

        # Log parsed fields to txt (raw already logged inside llm.complete)
        self.llm.log_parsed(f"{self.agent_id} tick={tick} harvest_decision", parsed)

        # Store memories
        self._store_memories(parsed, observation, tick)

        return parsed

    def _format_others_harvests(
        self,
        others_harvests: dict,
        my_last_harvest,
        tick: int,
        observation: dict = None,
    ) -> str:
        if not others_harvests and my_last_harvest is None:
            return "No harvest data yet — this is the first season."

        # Use the label passed in via observation (computed from actual history),
        # falling back to tick arithmetic only if not provided.
        if observation and observation.get("last_season_label"):
            season_label = observation["last_season_label"]
        else:
            from core.environment import FisheryEnvironment as _FE
            prev_tick    = tick - 1
            season_label = _FE.season_label(prev_tick) if prev_tick >= 0 else "Last season"

        lines = ["Last season's harvests:", f"  {season_label}"]

        if others_harvests:
            # Include own harvest in max calculation so ← highest is fair
            all_vals = list(others_harvests.values())
            if my_last_harvest is not None:
                all_vals.append(my_last_harvest)
            max_val = max(all_vals) if all_vals else 0

            for aid, amt in sorted(others_harvests.items()):
                marker = "  \u2190 highest" if amt == max_val and amt > 0 else ""
                lines.append(f"  {aid}: {amt:.1f} units{marker}")

            total = sum(others_harvests.values())
            if my_last_harvest is not None:
                total += my_last_harvest
            lines.append(f"  Community total: {total:.1f} units")

        if my_last_harvest is not None:
            marker = "  \u2190 highest" if my_last_harvest == max_val and max_val > 0 else ""
            lines.append(f"  Your harvest: {my_last_harvest:.1f} units{marker}")

        # Stock status line with actual numbers
        if observation:
            desc      = observation.get("stock_description", "")
            trend     = observation.get("stock_trend", "")
            stock     = observation.get("stock")
            max_stock = observation.get("max_stock")
            if desc:
                trend_tag = f" ({trend})" if trend and trend != "unknown" else ""
                if stock is not None and max_stock:
                    lines.append(
                        f"  Current stock: {stock:.1f} / {max_stock:.1f} units — "
                        f"{desc}{trend_tag}"
                    )
                else:
                    lines.append(f"  {desc}{trend_tag}")

        return "\n".join(lines)

    def _parse_cot(self, raw: str, tick: int) -> dict:
        """Parse all 8 fields from CoT output.

        Handles both plain and markdown-bold headers:
          LABEL: content
          **LABEL**        (content on next line)
          **LABEL:**       (content on next line)
        """
        # Next-section lookahead: either **LABEL** or LABEL: (3+ uppercase/underscore chars)
        _NEXT = r"(?=\n[ \t]*(?:\*{1,2}[A-Z_]{3,}|\b[A-Z_]{3,}:)|\Z)"

        def extract(label: str, text: str) -> str:
            esc = re.escape(label)
            # Header: **LABEL**, **LABEL:**, **LABEL**: or plain LABEL:
            header = rf"(?:\*{{1,2}}{esc}\*{{1,2}}:?|{esc}:)"
            pattern = header + r"[ \t]*\n?\s*(.*?)\s*" + _NEXT
            m = re.search(pattern, text, re.DOTALL)
            if m:
                return m.group(1).strip()
            return ""

        stock_assessment   = extract("STOCK_ASSESSMENT",   raw)
        others_behaviour   = extract("OTHERS_BEHAVIOUR",   raw)
        long_term_thinking = extract("LONG_TERM_THINKING", raw)
        norm_active_raw    = extract("NORM_ACTIVE",        raw).lower()
        norm_content       = extract("NORM_CONTENT",       raw)
        harvest_reasoning  = extract("HARVEST_REASONING",  raw)
        harvest_amount_raw = extract("HARVEST_AMOUNT",     raw)
        what_you_might_say = extract("WHAT_YOU_MIGHT_SAY", raw)

        # Normalise norm_active
        if "yes" in norm_active_raw:
            norm_active = "yes"
        elif "forming" in norm_active_raw:
            norm_active = "forming"
        else:
            norm_active = "no"

        # Parse harvest amount — first float found, clamped to [0, 10]
        harvest_amount = 5.0
        nums = re.findall(r"\d+\.?\d*", harvest_amount_raw)
        if nums:
            try:
                harvest_amount = float(nums[0])
            except ValueError:
                pass
        harvest_amount = max(0.0, min(10.0, harvest_amount))

        # Clean what_you_might_say
        say = what_you_might_say.strip()
        if say.lower() in {"nothing", "none", ""}:
            say = "nothing"

        return {
            "stock_assessment":   stock_assessment,
            "others_behaviour":   others_behaviour,
            "long_term_thinking": long_term_thinking,
            "norm_active":        norm_active,
            "norm_content":       norm_content,
            "harvest_reasoning":  harvest_reasoning,
            "harvest_amount":     harvest_amount,
            "what_you_might_say": say,
        }

    def _store_memories(self, parsed: dict, observation: dict, tick: int):
        """Store observations, own decision, and norm belief in memory."""
        # Others' harvests as individual observations
        for other_id, amount in observation.get("others_harvests", {}).items():
            self.memory.add(
                content=f"{other_id} harvested {amount:.1f} units (season {tick})",
                importance=4,
                memory_type="observation",
                tick=tick,
            )

        # Own decision
        reasoning_snippet = parsed["harvest_reasoning"][:120]
        self.memory.add(
            content=(
                f"Season {tick}: I harvested {parsed['harvest_amount']:.1f} units. "
                f"Reasoning: {reasoning_snippet}"
            ),
            importance=7,
            memory_type="harvest_decision",
            tick=tick,
        )

        # Norm belief
        if parsed["norm_active"] != "no":
            norm_snippet = parsed["norm_content"][:500]
            self.memory.add(
                content=(
                    f"Season {tick}: I believed a norm was {parsed['norm_active']}: "
                    f"'{norm_snippet}'"
                ),
                importance=8,
                memory_type="norm_belief",
                tick=tick,
            )

    def run_reflection(self, current_tick: int):
        """Separate LLM call. Only when memory.should_reflect() is True."""
        recent = self.memory.get_recent(n=20)
        if not recent:
            return

        recent_text = "\n".join(
            f"  - [tick {m.created_tick}] {m.content}" for m in recent
        )

        prompt = f"""You are {self.agent_id}, a fisher reflecting on recent seasons.

{self.persona}

Your recent experiences:
{recent_text}

Based only on these experiences, what are your 3 most important insights about:
1. The current state and trajectory of the fish stock
2. What other fishers in this community tend to do
3. Whether any shared norms or understandings are forming

Format each insight as:
INSIGHT: [your insight] (supported by: [brief reference to memories])"""

        raw = self.llm.complete(
            prompt, max_tokens=2000,
            label=f"reflection {self.agent_id} tick={current_tick}",
        )

        # Parse INSIGHT lines
        insights = re.findall(
            r"INSIGHT:\s*(.+?)(?=INSIGHT:|$)", raw, re.DOTALL
        )
        parsed_insights = {f"insight_{i+1}": s.strip() for i, s in enumerate(insights) if s.strip()}
        self.llm.log_parsed(f"{self.agent_id} tick={current_tick} reflection", parsed_insights)

        for insight in insights:
            insight = insight.strip()
            if insight:
                self.memory.add(
                    content=insight,
                    importance=9,
                    memory_type="reflection",
                    tick=current_tick,
                )
