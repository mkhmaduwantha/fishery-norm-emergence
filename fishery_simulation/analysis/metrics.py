"""
Quantitative metrics computed from logged records.
All functions take list[dict] (loaded from JSONL).
"""


def stock_trajectory(records: list) -> list:
    """[stock_pct per tick]"""
    return [r["stock_pct"] for r in records]


def mean_harvest_over_time(records: list) -> list:
    """[mean_harvest per tick]"""
    return [r["mean_harvest"] for r in records]


def harvest_variance_over_time(records: list) -> list:
    """[harvest_variance per tick] — high variance = uncoordinated"""
    return [r["harvest_variance"] for r in records]


def norm_adoption_trajectory(records: list) -> list:
    """[norm_adoption_rate per tick]"""
    return [r["norm_adoption_rate"] for r in records]


def sustainability_score(records: list, n_ticks: int) -> float:
    """
    Single number summarising run outcome. Range [0, 1].
    Not collapsed: final_stock_pct
    Collapsed: collapse_tick / n_ticks
    """
    if not records:
        return 0.0
    final = records[-1]
    if final.get("collapsed", False):
        for r in records:
            if r.get("collapsed", False):
                return r["tick"] / n_ticks if n_ticks > 0 else 0.0
        return 0.0
    return final["stock_pct"]


def compare_conditions(results_by_condition: dict) -> dict:
    """
    results_by_condition = {"baseline": [summaries], "long_term": [...]}

    For each condition compute aggregated stats across replications.
    """
    comparison = {}

    for condition, summaries in results_by_condition.items():
        if not summaries:
            continue

        n = len(summaries)

        def safe_mean(values):
            vals = [v for v in values if v is not None]
            return sum(vals) / len(vals) if vals else None

        scores          = [s.get("sustainability_score", 0.0) for s in summaries]
        collapsed_flags = [1 if s.get("collapsed", False) else 0 for s in summaries]
        norm_peaks      = [s.get("peak_norm_adoption", 0.0) for s in summaries]
        first_proposals = [s.get("first_proposal_tick") for s in summaries]
        total_agr       = [s.get("total_agreements", 0) for s in summaries]

        comparison[condition] = {
            "n_replications":            n,
            "mean_sustainability_score": safe_mean(scores),
            "collapse_rate":             sum(collapsed_flags) / n,
            "mean_peak_norm_adoption":   safe_mean(norm_peaks),
            "mean_first_proposal_tick":  safe_mean(first_proposals),
            "mean_total_agreements":     safe_mean(total_agr),
        }

    return comparison
