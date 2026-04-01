"""
Orchestrates the full simulation using a LangGraph state machine.

Block structure (configurable via seasons_per_reflection):
  Repeat N times:
    Phase A — all agents harvest simultaneously (or skip if fishery closed)
    Phase B — apply harvests, update environment
    Phase C — communication window (condition-dependent; skip if closed)
    Phase E — log tick, advance counters
  Then once:
    Phase D — every agent reflects (one call each)
  → END

There is no early termination on stock collapse. A collapse triggers a
2-season closed period recorded in history; then stock resets and
fishing resumes. The simulation always completes its full block.
"""
import os
import random
from typing import TypedDict, Any

from langgraph.graph import StateGraph, END, START

from core.environment import FisheryEnvironment
from core.agent import FisheryAgent, PERSONAS
from core.llm import LLMAdapter
from core.dialogue import DialogueEngine
from experiments.conditions import CONDITIONS_WITH_COMMUNICATION
from experiments.logger import ExperimentLogger


# ─────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────

def pair_communicators(
    decisions: dict,
    interaction_counts: dict,
    all_agent_ids: list,
) -> list:
    """
    Returns (initiator, responder) pairs for this tick.
    Each agent appears at most once. At most N//2 pairs.
    Weighted by inverse prior-interaction count.
    """
    would_say = [
        aid for aid, cot in decisions.items()
        if cot.get("what_you_might_say", "nothing").strip().lower()
        not in {"nothing", "none", ""}
    ]
    random.shuffle(would_say)

    used, pairs = set(), []
    for initiator_id in would_say:
        if initiator_id in used:
            continue
        candidates = [
            aid for aid in all_agent_ids
            if aid != initiator_id and aid not in used
        ]
        if not candidates:
            break
        weights = [
            1.0 / (1 + interaction_counts.get(initiator_id, {}).get(c, 0))
            for c in candidates
        ]
        responder_id = random.choices(candidates, weights=weights, k=1)[0]
        pairs.append((initiator_id, responder_id))
        used.add(initiator_id)
        used.add(responder_id)

    return pairs


def identify_conversation_type(turns: list) -> str:
    """Classify conversation by its speech act sequence."""
    acts = [t.speech_act for t in turns]

    if "proposal" in acts:
        idx   = acts.index("proposal")
        after = acts[idx + 1:]
        if "agreement" in after:
            return "norm_negotiated_accepted"
        elif "disagreement" in after:
            return "norm_negotiated_rejected"
        else:
            return "norm_proposed_unresolved"

    if "warning" in acts:
        idx   = acts.index("warning")
        after = acts[idx + 1:]
        if "agreement" in after:
            return "warning_accepted"
        elif "disagreement" in after:
            return "warning_defied"
        else:
            return "warning_acknowledged"

    return "general_conversation"


# ─────────────────────────────────────────────────────────────────
# LangGraph state
# ─────────────────────────────────────────────────────────────────

class SimState(TypedDict):
    # ── Configuration (set once) ──────────────────────────────────
    seasons_per_reflection: int
    condition: str
    enable_communication: bool

    # ── Counters ──────────────────────────────────────────────────
    tick: int              # absolute season number (ever-incrementing)
    season_in_block: int   # how many seasons completed in current block

    # ── Mutable simulation objects ────────────────────────────────
    agents: Any            # dict[str, FisheryAgent]
    env: Any               # FisheryEnvironment
    dialogue_engine: Any   # DialogueEngine
    logger: Any            # ExperimentLogger

    # ── Per-season data ───────────────────────────────────────────
    world_nl: str
    decisions: dict
    actual_yields: dict
    last_harvests: dict
    last_season_label: str
    interaction_counts: dict
    dialogue_records: list


# ─────────────────────────────────────────────────────────────────
# Phase nodes
# ─────────────────────────────────────────────────────────────────

def phase_a_harvest(state: SimState) -> dict:
    """
    Phase A: All agents decide simultaneously.

    If the fishery is in a closed season (recovering from collapse),
    agents still observe the situation but submit 0.0 harvests — their
    CoT reasoning will reflect on the collapse and the closed fishery.
    The environment's apply_harvests() handles 0-yield correctly.
    """
    env           = state["env"]
    agents        = state["agents"]
    tick          = state["tick"]
    last_harvests = state["last_harvests"]

    world_nl           = env.natural_language_state()
    harvest_history_nl = env.format_harvest_history(last_n=5)
    decisions          = {}

    last_season_label = state.get("last_season_label", "")

    for agent_id, agent in agents.items():
        obs = env.get_observation_for(agent_id, last_harvests)
        obs["harvest_history_nl"]  = harvest_history_nl   # injected here
        obs["last_season_label"]   = last_season_label
        cot = agent.decide(obs, world_nl)
        decisions[agent_id] = cot

    return {"world_nl": world_nl, "decisions": decisions}


def phase_b_apply(state: SimState) -> dict:
    """Phase B: Apply all harvests simultaneously, update world."""
    env       = state["env"]
    decisions = state["decisions"]

    harvests      = {aid: cot["harvest_amount"] for aid, cot in decisions.items()}
    actual_yields = env.apply_harvests(harvests)

    # Derive label from the last actual (non-closed) history entry so that
    # after a mid-year collapse the label matches the data (not tick-1).
    history     = env.state.harvest_history
    last_actual = next(
        (h for h in reversed(history) if not h.get("closed_season", False)),
        None,
    )
    if last_actual is not None:
        last_season_label = FisheryEnvironment.season_label(last_actual["tick"])
        # Use actual yields from that entry so amounts match the label
        last_harvests = {
            aid: last_actual["actual"].get(aid, 0.0) for aid in harvests
        }
    else:
        last_season_label = ""
        last_harvests = harvests

    return {
        "actual_yields":    actual_yields,
        "last_harvests":    last_harvests,
        "last_season_label": last_season_label,
    }


def phase_c_communicate(state: SimState) -> dict:
    """
    Phase C: Communication window.
    Skipped during closed seasons — no communal fishing activity.
    """
    env = state["env"]

    # Skip if fishery was collapsed BEFORE this phase-b applied
    # (env.state.collapsed reflects post-harvest state of NEXT tick now,
    #  so check via last_harvests being all-zero as proxy)
    last = state.get("last_harvests", {})
    fishery_was_closed = all(v == 0.0 for v in last.values()) if last else False

    if not state["enable_communication"] or fishery_was_closed:
        return {"dialogue_records": []}

    agents             = state["agents"]
    dialogue_engine    = state["dialogue_engine"]
    decisions          = state["decisions"]
    interaction_counts = state["interaction_counts"]
    world_nl           = state["world_nl"]
    tick               = state["tick"]

    harvest_history_nl = env.format_harvest_history(last_n=8)
    pairs              = pair_communicators(decisions, interaction_counts, list(agents.keys()))
    dialogue_records   = []

    for initiator_id, responder_id in pairs:
        opening = decisions[initiator_id]["what_you_might_say"]
        if not opening or opening.strip().lower() in {"nothing", "none"}:
            continue

        turns = dialogue_engine.run(
            initiator_id=initiator_id,
            responder_id=responder_id,
            initiator_memory=agents[initiator_id].memory,
            responder_memory=agents[responder_id].memory,
            initiator_persona=agents[initiator_id].persona,
            responder_persona=agents[responder_id].persona,
            opening_message=opening,
            world_nl=world_nl,
            tick=tick,
            max_turns=6,
            harvest_history_nl=harvest_history_nl,
        )

        dialogue_records.append({
            "tick":              tick,
            "initiator":         initiator_id,
            "responder":         responder_id,
            "turns": [
                {
                    "speaker_id":  t.speaker_id,
                    "listener_id": t.listener_id,
                    "content":     t.content,
                    "speech_act":  t.speech_act,
                    "turn_number": t.turn_number,
                }
                for t in turns
            ],
            "conversation_type": identify_conversation_type(turns),
            "n_turns":           len(turns),
        })

        interaction_counts.setdefault(initiator_id, {})
        prev = interaction_counts[initiator_id].get(responder_id, 0)
        interaction_counts[initiator_id][responder_id] = prev + 1

    return {
        "dialogue_records":   dialogue_records,
        "interaction_counts": interaction_counts,
    }


def phase_d_reflect(state: SimState) -> dict:
    """
    Phase D: One reflection call for every agent.
    Always fires exactly once at end of the block.
    """
    tick = state["tick"]
    for agent in state["agents"].values():
        agent.run_reflection(tick)
    return {}


def phase_e_log(state: SimState) -> dict:
    """Phase E: Log season data, advance counters."""
    logger = state["logger"]
    env    = state["env"]
    tick   = state["tick"]

    logger.log_tick(
        tick=tick,
        env_state=env.state,
        decisions=state["decisions"],
        actual_yields=state["actual_yields"],
        dialogue_records=state["dialogue_records"],
    )

    return {
        "tick":            tick + 1,
        "season_in_block": state["season_in_block"] + 1,
    }


def _after_season(state: SimState) -> str:
    """
    Route after logging a season.
      "reflect" — block complete → run reflection then END
      "more"    — continue to next season in this block
    No early-termination on collapse; the environment handles recovery.
    """
    if state["season_in_block"] >= state["seasons_per_reflection"]:
        return "reflect"
    return "more"


# ─────────────────────────────────────────────────────────────────
# Build the LangGraph graph
#
#   START → phase_a → phase_b → phase_c → phase_e → _after_season?
#                                                       "more"    → phase_a
#                                                       "reflect" → phase_d → END
# ─────────────────────────────────────────────────────────────────

def _build_graph():
    g = StateGraph(SimState)

    g.add_node("phase_a", phase_a_harvest)
    g.add_node("phase_b", phase_b_apply)
    g.add_node("phase_c", phase_c_communicate)
    g.add_node("phase_d", phase_d_reflect)
    g.add_node("phase_e", phase_e_log)

    g.add_edge(START,     "phase_a")
    g.add_edge("phase_a", "phase_b")
    g.add_edge("phase_b", "phase_c")
    g.add_edge("phase_c", "phase_e")
    g.add_conditional_edges(
        "phase_e",
        _after_season,
        {"more": "phase_a", "reflect": "phase_d"},
    )
    g.add_edge("phase_d", END)

    return g.compile()


# ─────────────────────────────────────────────────────────────────
# SimulationRunner
# ─────────────────────────────────────────────────────────────────

class SimulationRunner:
    def __init__(
        self,
        condition: str,
        n_agents: int = 8,
        seasons_per_reflection: int = 5,
        n_replications: int = 1,
        llm_provider: str = "ollama",
        llm_model: str = None,
        output_dir: str = "results",
    ):
        self.condition              = condition
        self.n_agents               = n_agents
        self.seasons_per_reflection = seasons_per_reflection
        self.n_replications         = n_replications
        self.enable_communication   = condition in CONDITIONS_WITH_COMMUNICATION
        self.llm_provider           = llm_provider
        self.llm_model              = llm_model
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def run_all(self) -> list:
        results = []
        for rep in range(self.n_replications):
            print(
                f"  [{self.condition}] rep {rep + 1}/{self.n_replications} "
                f"({self.seasons_per_reflection} seasons → reflect → stop)"
            )
            results.append(self.run_replication(rep))
        return results

    def run_replication(self, rep_id: int) -> dict:
        llm             = LLMAdapter(
            provider=self.llm_provider,
            model=self.llm_model,
            log_dir=os.path.join(self.output_dir, "logs"),
        )
        dialogue_engine = DialogueEngine(llm)
        env             = FisheryEnvironment(n_agents=self.n_agents)

        personas = random.sample(PERSONAS, min(self.n_agents, len(PERSONAS)))
        while len(personas) < self.n_agents:
            personas.append(random.choice(PERSONAS))

        agents = {
            f"agent_{i}": FisheryAgent(
                agent_id=f"agent_{i}",
                persona=personas[i],
                condition=self.condition,
                llm=llm,
            )
            for i in range(self.n_agents)
        }

        logger    = ExperimentLogger(self.condition, rep_id)
        sim_graph = _build_graph()

        initial_state: SimState = {
            "seasons_per_reflection": self.seasons_per_reflection,
            "condition":              self.condition,
            "enable_communication":   self.enable_communication,
            "tick":                   0,
            "season_in_block":        0,
            "agents":                 agents,
            "env":                    env,
            "dialogue_engine":        dialogue_engine,
            "logger":                 logger,
            "world_nl":               "",
            "decisions":              {},
            "actual_yields":          {},
            "last_harvests":          {},
            "last_season_label":      "",
            "interaction_counts":     {},
            "dialogue_records":       [],
        }

        sim_graph.invoke(initial_state)

        filepath = os.path.join(
            self.output_dir,
            f"{self.condition}_rep{rep_id}.jsonl",
        )
        logger.save(filepath)

        usage = llm.usage_summary()
        llm.close()
        print(
            f"    API calls: {usage['total_calls']} | "
            f"tokens: {usage['total_input_tokens']}in "
            f"{usage['total_output_tokens']}out"
        )
        return logger.summarise()
