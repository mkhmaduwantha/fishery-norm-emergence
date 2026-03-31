"""
Experimental conditions. These are the ONLY differences between runs.
Never mention what others do, what norms exist, or what agents should believe.
"""

CONDITION_PROMPTS = {
    "baseline": "",

    "long_term": (
        "You have fished this stock for many years and expect "
        "to continue for the rest of your working life. "
        "Your children may fish here after you."
    ),

    "reputational": (
        "You are a known member of this fishing community. "
        "Your harvest history is visible to all other fishers, "
        "as theirs is to you. Your reputation as a responsible "
        "community member is at stake in how you fish."
    ),

    "communication": "",
    # Communication is structural — enabled in runner, not prompt.

    "full_social": (
        "You are a known member of this fishing community. "
        "Your harvest history is visible to all other fishers, "
        "as theirs is to you."
    ),
    # full_social = reputational framing + communication enabled.
}

CONDITIONS_WITH_COMMUNICATION = {"communication", "full_social"}
