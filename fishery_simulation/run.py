#!/usr/bin/env python3
"""
Fishery commons norm emergence simulation.

Usage examples:
  # Quick smoke test: 3 agents, 3 seasons, then reflect, then stop
  python run.py --condition baseline --agents 3 --seasons 3

  # Dry run — print one agent prompt without any LLM calls
  python run.py --condition baseline --dry-run

  # 8 agents, 5 seasons per block, 1 replication
  python run.py --condition full_social --agents 8 --seasons 5

  # More seasons before reflection
  python run.py --condition communication --agents 8 --seasons 10 --reps 3

  # Run all conditions
  python run.py --all-conditions --agents 8 --seasons 5

  # Use Anthropic or OpenAI instead of Ollama
  python run.py --provider anthropic --condition baseline --agents 3 --seasons 3

LLM call log: results/logs/llm_calls.jsonl  (prompt + response per line)
Sim results:  results/<condition>_rep<N>.jsonl
"""

import argparse
import json
import os
from dotenv import load_dotenv

load_dotenv()

CONDITIONS = [
    "baseline", "long_term", "reputational", "communication", "full_social"
]


def dry_run(condition: str, n_agents: int):
    """Print a complete agent_0 prompt at tick 0 without making any LLM calls."""
    from core.environment import FisheryEnvironment
    from core.agent import PERSONAS
    from core.memory import MemoryStream
    from experiments.conditions import CONDITION_PROMPTS

    env = FisheryEnvironment(n_agents=n_agents)

    # Simulate tick-0 state with fake previous harvests
    fake_last = {f"agent_{i}": 5.0 for i in range(n_agents)}
    obs      = env.get_observation_for("agent_0", fake_last)
    world_nl = env.natural_language_state()
    persona  = PERSONAS[0]
    cond_ins = CONDITION_PROMPTS.get(condition, "")

    # Build others-harvests block exactly as agent.decide() would
    others = {k: v for k, v in fake_last.items() if k != "agent_0"}
    if others:
        max_val = max(others.values())
        lines = ["Last season's harvests:"]
        for aid, amt in sorted(others.items()):
            marker = "  \u2190 highest" if amt == max_val else ""
            lines.append(f"  {aid}: {amt:.1f} units{marker}")
        total = sum(others.values()) + 5.0
        lines.append(f"Community total: {total:.1f} units")
        lines.append("Your harvest: 5.0 units")
        others_text = "\n".join(lines)
    else:
        others_text = "No harvest data yet — this is the first season."

    memory    = MemoryStream()
    memories  = memory.retrieve(query="stock harvest season", top_k=6, current_tick=0)
    mem_text  = memory.format_for_prompt(memories)

    prompt = f"""You are agent_0, a fisher who depends on a shared
fish stock for your livelihood.

{persona}

Current situation:
{world_nl}

{others_text}

{cond_ins}

What you remember from past seasons:
{mem_text}

Reason through your decision step by step:

STOCK_ASSESSMENT: What is your reading of the stock right now?
OTHERS_BEHAVIOUR: What pattern do you notice in what others are taking? Is anyone taking significantly more or less?
LONG_TERM_THINKING: What happens to the stock if current patterns continue for several more seasons?
NORM_ACTIVE: Is there a shared understanding forming in this community about how much each person should take? Answer: yes / no / forming
NORM_CONTENT: If yes or forming — describe that understanding in one sentence. If no — write "none".
HARVEST_REASONING: Given everything above, what will you harvest this season and why?
HARVEST_AMOUNT: [a number between 0.0 and 10.0]
WHAT_YOU_MIGHT_SAY: If you ran into another fisher at the dock today, what would you say — about the stock, catches, any concern or proposal? Write "nothing" if you have nothing to say."""

    print("=" * 70)
    print("DRY RUN — agent_0 prompt at tick 0")
    print("=" * 70)
    print(f"Condition : {condition}")
    print(f"N agents  : {n_agents}")
    print()
    print("WORLD STATE:")
    print(world_nl)
    print()
    print("OBSERVATION (logged fields):")
    # Remove stock_pct from display since it's log-only
    obs_display = {k: v for k, v in obs.items() if k != "stock_pct"}
    print(json.dumps(obs_display, indent=2))
    print()
    print("CONDITION INSERT:")
    print(cond_ins if cond_ins else "(empty — baseline)")
    print()
    print("FULL PROMPT:")
    print("-" * 70)
    print(prompt)
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Fishery commons norm emergence simulation"
    )
    parser.add_argument(
        "--condition", default="baseline", choices=CONDITIONS,
        help="Experimental condition"
    )
    parser.add_argument(
        "--all-conditions", action="store_true",
        help="Run all 5 conditions in sequence"
    )
    parser.add_argument("--agents",  type=int, default=8)
    parser.add_argument(
        "--seasons", type=int, default=5,
        help="Seasons per block before reflection (default 5)"
    )
    parser.add_argument("--reps",    type=int, default=1)
    parser.add_argument(
        "--provider", default="ollama",
        choices=["ollama", "anthropic", "openai"],
        help="LLM provider (default: ollama)"
    )
    parser.add_argument(
        "--model", default=None,
        help="Override model name (e.g. gpt-oss:20b)"
    )
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print prompt without making LLM calls"
    )
    args = parser.parse_args()

    if args.dry_run:
        dry_run(args.condition, args.agents)
        return

    from experiments.runner import SimulationRunner
    from analysis.metrics import compare_conditions

    conditions_to_run = CONDITIONS if args.all_conditions else [args.condition]

    all_results = {}
    for condition in conditions_to_run:
        print(f"\nRunning condition: {condition}")
        runner = SimulationRunner(
            condition=condition,
            n_agents=args.agents,
            seasons_per_reflection=args.seasons,
            n_replications=args.reps,
            llm_provider=args.provider,
            llm_model=args.model,
            output_dir=args.output_dir,
        )
        summaries = runner.run_all()
        all_results[condition] = summaries

        print(f"\n  {condition} results:")
        for s in summaries:
            print(
                f"    rep {s['replication']}: "
                f"stock={s['final_stock_pct']:.1f}% "
                f"collapsed={s['collapsed']} "
                f"norm_peak={s['peak_norm_adoption']:.2f} "
                f"agreements={s['total_agreements']}"
            )

    if len(conditions_to_run) > 1:
        print("\nCROSS-CONDITION COMPARISON:")
        comparison = compare_conditions(all_results)
        print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
