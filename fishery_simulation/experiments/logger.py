"""
Structured logging to JSONL. One record per tick.
"""
import json
import statistics


class ExperimentLogger:
    def __init__(self, condition: str, rep_id: int):
        self.condition = condition
        self.rep_id = rep_id
        self.records = []
        self.collapse_tick = None

    def log_tick(
        self,
        tick: int,
        env_state,
        decisions: dict,
        actual_yields: dict,
        dialogue_records: list,
    ):
        harvests = {aid: cot["harvest_amount"] for aid, cot in decisions.items()}
        total_harvest = sum(actual_yields.values())
        harvest_values = list(harvests.values())
        mean_harvest = (
            sum(harvest_values) / len(harvest_values) if harvest_values else 0.0
        )
        variance = (
            statistics.variance(harvest_values)
            if len(harvest_values) > 1
            else 0.0
        )

        stock_pct = (
            env_state.stock / env_state.max_stock
            if env_state.max_stock > 0
            else 0.0
        )

        # norm_adoption_rate: fraction where norm_active != "no"
        norm_count = sum(
            1 for cot in decisions.values()
            if cot.get("norm_active", "no") != "no"
        )
        norm_adoption_rate = norm_count / len(decisions) if decisions else 0.0

        # dialogue stats
        n_conversations = len(dialogue_records)
        n_proposals = sum(
            1
            for rec in dialogue_records
            for turn in rec.get("turns", [])
            if turn.get("speech_act") == "proposal"
        )
        n_agreements = sum(
            1
            for rec in dialogue_records
            for turn in rec.get("turns", [])
            if turn.get("speech_act") == "agreement"
        )
        n_warnings = sum(
            1
            for rec in dialogue_records
            for turn in rec.get("turns", [])
            if turn.get("speech_act") == "warning"
        )

        record = {
            "tick": tick,
            "condition": self.condition,
            "replication": self.rep_id,
            "stock_after": env_state.stock,
            "stock_pct": stock_pct,
            "collapsed": env_state.collapsed,
            "harvests": harvests,
            "actual_yields": actual_yields,
            "total_harvest": total_harvest,
            "mean_harvest": mean_harvest,
            "harvest_variance": variance,
            "cot_outputs": {
                aid: {
                    "stock_assessment":   cot.get("stock_assessment", ""),
                    "others_behaviour":   cot.get("others_behaviour", ""),
                    "long_term_thinking": cot.get("long_term_thinking", ""),
                    "norm_active":        cot.get("norm_active", "no"),
                    "norm_content":       cot.get("norm_content", "none"),
                    "harvest_reasoning":  cot.get("harvest_reasoning", ""),
                    "harvest_amount":     cot.get("harvest_amount", 5.0),
                    "what_you_might_say": cot.get("what_you_might_say", "nothing"),
                }
                for aid, cot in decisions.items()
            },
            "dialogue_records": dialogue_records,
            "norm_adoption_rate": norm_adoption_rate,
            "n_conversations": n_conversations,
            "n_proposals": n_proposals,
            "n_agreements": n_agreements,
            "n_warnings": n_warnings,
        }
        self.records.append(record)

    def log_collapse(self, tick: int):
        self.collapse_tick = tick

    def save(self, filepath: str):
        """Write each record as one JSON line (indent=2)."""
        with open(filepath, "w") as f:
            for record in self.records:
                f.write(json.dumps(record, indent=2) + "\n")

    def summarise(self) -> dict:
        """Compute summary statistics across all ticks."""
        if not self.records:
            return {
                "condition": self.condition,
                "replication": self.rep_id,
                "n_ticks_run": 0,
                "collapsed": False,
                "collapse_tick": None,
                "final_stock_pct": 0.0,
                "mean_harvest_all": 0.0,
                "mean_harvest_last5": 0.0,
                "harvest_variance_mean": 0.0,
                "peak_norm_adoption": 0.0,
                "ticks_majority_norm": 0,
                "first_proposal_tick": None,
                "first_agreement_tick": None,
                "first_warning_tick": None,
                "total_conversations": 0,
                "total_proposals": 0,
                "total_agreements": 0,
                "total_warnings": 0,
                "sustainability_score": 0.0,
            }

        n_ticks_run = len(self.records)
        collapsed = self.records[-1].get("collapsed", False) or (
            self.collapse_tick is not None
        )
        final_stock_pct = self.records[-1]["stock_pct"]

        all_means = [r["mean_harvest"] for r in self.records]
        mean_harvest_all = sum(all_means) / len(all_means)

        last5 = [r["mean_harvest"] for r in self.records[-5:]]
        mean_harvest_last5 = sum(last5) / len(last5) if last5 else 0.0

        variances = [r["harvest_variance"] for r in self.records]
        harvest_variance_mean = sum(variances) / len(variances) if variances else 0.0

        norm_rates = [r["norm_adoption_rate"] for r in self.records]
        peak_norm_adoption = max(norm_rates) if norm_rates else 0.0
        ticks_majority_norm = sum(1 for r in norm_rates if r > 0.5)

        first_proposal_tick = None
        first_agreement_tick = None
        first_warning_tick = None
        total_conversations = 0
        total_proposals = 0
        total_agreements = 0
        total_warnings = 0

        for r in self.records:
            total_conversations += r["n_conversations"]
            total_proposals += r["n_proposals"]
            total_agreements += r["n_agreements"]
            total_warnings += r["n_warnings"]

            if first_proposal_tick is None and r["n_proposals"] > 0:
                first_proposal_tick = r["tick"]
            if first_agreement_tick is None and r["n_agreements"] > 0:
                first_agreement_tick = r["tick"]
            if first_warning_tick is None and r["n_warnings"] > 0:
                first_warning_tick = r["tick"]

        # Sustainability score
        if collapsed:
            ct = self.collapse_tick or n_ticks_run
            sustainability_score = ct / n_ticks_run if n_ticks_run > 0 else 0.0
        else:
            sustainability_score = final_stock_pct

        return {
            "condition": self.condition,
            "replication": self.rep_id,
            "n_ticks_run": n_ticks_run,
            "collapsed": collapsed,
            "collapse_tick": self.collapse_tick,
            "final_stock_pct": final_stock_pct,
            "mean_harvest_all": mean_harvest_all,
            "mean_harvest_last5": mean_harvest_last5,
            "harvest_variance_mean": harvest_variance_mean,
            "peak_norm_adoption": peak_norm_adoption,
            "ticks_majority_norm": ticks_majority_norm,
            "first_proposal_tick": first_proposal_tick,
            "first_agreement_tick": first_agreement_tick,
            "first_warning_tick": first_warning_tick,
            "total_conversations": total_conversations,
            "total_proposals": total_proposals,
            "total_agreements": total_agreements,
            "total_warnings": total_warnings,
            "sustainability_score": sustainability_score,
        }
