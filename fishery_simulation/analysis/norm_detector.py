"""
Detects Ostrom norm types in agent reasoning and dialogue.
Called on logged data — never on live agents.
"""
import re


NORM_PATTERNS = {
    "restraint": [
        r"take (only|less|not too much|moderate|reasonable)",
        r"leave (enough|some) for (others|future|next)",
        r"sustainabl",
        r"don.t (over|exhaust|deplete|take too much)",
        r"careful(ly)? about how much",
        r"hold back",
    ],
    "quota": [
        r"(each|everyone|all of us) (should|must|ought to) take",
        r"limit (of|to) \d",
        r"no more than \d",
        r"(per.person|per.fisher|individual) (quota|limit|share|cap)",
        r"agree(d)? (on|to) \d",
        r"(maximum|max).{0,10}\d",
        r"4 units|3 units|5 units",
    ],
    "monitoring": [
        r"(watch|observe|monitor|track) (each other|what others)",
        r"(report|share|disclose).{0,20}catch",
        r"(transparent|openly|honestly|accountab)",
        r"(everyone|all) (know|see|report)",
    ],
    "sanction": [
        r"(warn|warning|consequence|sanction)",
        r"(exclude|ostracis|reputation)",
        r"(won.t|will not).{0,20}(cooperate|share|trust)",
        r"(shame|embarrass).{0,20}(community|others)",
        r"graduated|proportional",
    ],
    "reciprocity": [
        r"(I will|I.ll).{0,30}if (others|you|everyone)",
        r"(conditional|depends on|provided that)",
        r"(trust|rely).{0,20}others",
        r"if (others|they).{0,20}(same|their part|too)",
        r"tit.for.tat",
    ],
    "fairness": [
        r"(fair|equal|equit)",
        r"same (amount|rules|limit) for (all|everyone)",
        r"(no one|nobody) (gets|takes) more",
        r"(share|split).{0,15}(equal|fair)",
    ],
}


def detect_norms(text: str) -> dict:
    """Returns {norm_type: bool} for each type."""
    text_lower = text.lower()
    return {
        norm_type: any(re.search(p, text_lower) for p in patterns)
        for norm_type, patterns in NORM_PATTERNS.items()
    }


def analyse_run(records: list) -> dict:
    """
    For each tick, analyse CoT fields and dialogue for norm language.
    Returns norm emergence trajectories and Ostrom principle matches.
    """
    norm_types = list(NORM_PATTERNS.keys())
    norm_emergence_by_tick = {}
    norm_in_dialogue = []
    all_peaks = {nt: 0.0 for nt in norm_types}

    for record in records:
        tick = record["tick"]
        cot_outputs = record.get("cot_outputs", {})
        n_agents = max(len(cot_outputs), 1)

        tick_counts = {nt: 0 for nt in norm_types}

        for cot in cot_outputs.values():
            combined = " ".join([
                cot.get("others_behaviour",   ""),
                cot.get("long_term_thinking", ""),
                cot.get("norm_content",       ""),
                cot.get("harvest_reasoning",  ""),
            ])
            for nt, found in detect_norms(combined).items():
                if found:
                    tick_counts[nt] += 1

        fracs = {nt: tick_counts[nt] / n_agents for nt in norm_types}
        norm_emergence_by_tick[tick] = fracs

        for nt in norm_types:
            if fracs[nt] > all_peaks[nt]:
                all_peaks[nt] = fracs[nt]

        # Analyse dialogue
        for dlg in record.get("dialogue_records", []):
            all_text = " ".join(
                t.get("content", "") for t in dlg.get("turns", [])
            )
            found_norms = {
                nt: v for nt, v in detect_norms(all_text).items() if v
            }
            if found_norms:
                norm_in_dialogue.append({
                    "tick":              tick,
                    "initiator":         dlg.get("initiator"),
                    "responder":         dlg.get("responder"),
                    "norm_types":        list(found_norms.keys()),
                    "conversation_type": dlg.get("conversation_type"),
                })

    # first_appearance: first tick where fraction > 0.25
    first_appearance = {}
    for nt in norm_types:
        first_appearance[nt] = None
        for record in records:
            if norm_emergence_by_tick.get(record["tick"], {}).get(nt, 0) > 0.25:
                first_appearance[nt] = record["tick"]
                break

    ostrom_principles_matched = [
        nt for nt in norm_types if all_peaks[nt] > 0.4
    ]

    return {
        "norm_emergence_by_tick":    norm_emergence_by_tick,
        "first_appearance":          first_appearance,
        "norm_in_dialogue":          norm_in_dialogue,
        "ostrom_principles_matched": ostrom_principles_matched,
    }


def trace_punishment_effect(
    records: list,
    warned_agent: str,
    warning_tick: int,
) -> dict:
    """Compare warned agent's harvest before and after warning."""
    before_window = range(max(0, warning_tick - 3), warning_tick)
    after_window  = range(warning_tick + 1, warning_tick + 4)

    before_harvests, after_harvests = [], []

    for record in records:
        tick    = record["tick"]
        amount  = record.get("harvests", {}).get(warned_agent)
        if amount is None:
            continue
        if tick in before_window:
            before_harvests.append(amount)
        elif tick in after_window:
            after_harvests.append(amount)

    mean_before = sum(before_harvests) / len(before_harvests) if before_harvests else 0.0
    mean_after  = sum(after_harvests)  / len(after_harvests)  if after_harvests  else 0.0
    change      = mean_after - mean_before

    if change < -0.5:
        effect = "reduced"
    elif change > 0.5:
        effect = "increased"
    else:
        effect = "unchanged"

    return {
        "warned_agent": warned_agent,
        "warning_tick": warning_tick,
        "mean_before":  mean_before,
        "mean_after":   mean_after,
        "change":       change,
        "effect":       effect,
    }
