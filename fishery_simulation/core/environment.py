"""
Shared fish stock using logistic growth dynamics.
"""
from dataclasses import dataclass, field
from typing import Optional


DEFAULT_INITIAL_STOCK = 80.0
DEFAULT_MAX_STOCK = 100.0
DEFAULT_REGROWTH_RATE = 0.3
DEFAULT_COLLAPSE_THRESHOLD = 20.0
MAX_INDIVIDUAL_HARVEST = 10.0
SUSTAINABLE_TOTAL = 12.0  # reference value, not enforced


@dataclass
class FisheryState:
    stock: float
    max_stock: float
    regrowth_rate: float
    collapse_threshold: float
    collapsed: bool = False
    tick: int = 0
    harvest_history: list = field(default_factory=list)
    # Each entry: {tick, harvests{agent_id:amount}, total, stock_after}


class FisheryEnvironment:
    def __init__(
        self,
        n_agents: int = 8,
        initial_stock: float = DEFAULT_INITIAL_STOCK,
        max_stock: float = DEFAULT_MAX_STOCK,
        regrowth_rate: float = DEFAULT_REGROWTH_RATE,
        collapse_threshold: float = DEFAULT_COLLAPSE_THRESHOLD,
    ):
        self.n_agents = n_agents
        self.state = FisheryState(
            stock=initial_stock,
            max_stock=max_stock,
            regrowth_rate=regrowth_rate,
            collapse_threshold=collapse_threshold,
        )

    def apply_harvests(self, harvests: dict) -> dict:
        """
        Apply all agents' harvests simultaneously.
        If total > available stock: scale all harvests proportionally.
        Apply logistic regrowth.
        Check collapse threshold.
        Record to harvest_history.
        Increment tick.
        Return dict of actual yields (may be scaled).
        """
        s = self.state
        if s.collapsed:
            return {aid: 0.0 for aid in harvests}

        total_requested = sum(harvests.values())
        available = s.stock

        # Scale down if needed
        if total_requested > available and total_requested > 0:
            scale = available / total_requested
            actual = {aid: h * scale for aid, h in harvests.items()}
        else:
            actual = dict(harvests)

        total_actual = sum(actual.values())

        # Apply harvest
        s.stock = max(0.0, s.stock - total_actual)

        # Check collapse BEFORE regrowth
        if s.stock < s.collapse_threshold:
            s.collapsed = True
            s.stock = 0.0
            s.harvest_history.append({
                "tick": s.tick,
                "harvests": dict(harvests),
                "actual": actual,
                "total": total_actual,
                "stock_after": 0.0,
                "collapsed": True,
            })
            s.tick += 1
            return actual

        # Logistic regrowth
        growth = s.regrowth_rate * s.stock * (1 - s.stock / s.max_stock)
        s.stock = min(s.max_stock, s.stock + growth)

        s.harvest_history.append({
            "tick": s.tick,
            "harvests": dict(harvests),
            "actual": actual,
            "total": total_actual,
            "stock_after": s.stock,
            "collapsed": False,
        })
        s.tick += 1
        return actual

    def _stock_description(self) -> str:
        pct = self.state.stock / self.state.max_stock
        if pct > 0.75:
            return "The fish stock is abundant — fishing has been excellent."
        elif pct > 0.55:
            return "The stock is healthy but showing signs of pressure."
        elif pct > 0.35:
            return "The stock is under strain — catches per trip are down."
        elif pct > 0.20:
            return "The stock is in serious decline — the fishery is at risk."
        else:
            return "The stock is critically low — collapse is imminent."

    def _stock_trend(self) -> str:
        history = self.state.harvest_history
        if len(history) < 3:
            return "unknown"
        recent = history[-3:]
        stocks = [h["stock_after"] for h in recent]
        delta = stocks[-1] - stocks[0]
        if delta < -5:
            return "declining"
        elif delta > 3:
            return "recovering"
        else:
            return "stable"

    def get_observation_for(self, agent_id: str, last_harvests: dict) -> dict:
        """
        Returns observation dict for this agent.
        AGREED MONITORING: all harvests fully visible.
        """
        s = self.state
        pct = s.stock / s.max_stock if s.max_stock > 0 else 0.0

        others_harvests = {
            aid: amt for aid, amt in last_harvests.items()
            if aid != agent_id
        }
        my_last = last_harvests.get(agent_id, None)

        return {
            "stock_description": self._stock_description(),
            "stock_trend": self._stock_trend(),
            "stock_pct": pct,  # for logging only — not in prompt
            "others_harvests": others_harvests,
            "my_last_harvest": my_last,
            "tick": s.tick,
            "n_agents": self.n_agents,
            "collapsed": s.collapsed,
        }

    @staticmethod
    def _status_from_stock(stock_after: float, max_stock: float) -> str:
        """One-line status string from a historical stock level."""
        if max_stock <= 0:
            return "unknown"
        pct = stock_after / max_stock
        if pct > 0.75:
            return "abundant"
        elif pct > 0.55:
            return "healthy"
        elif pct > 0.35:
            return "under strain"
        elif pct > 0.20:
            return "in serious decline"
        else:
            return "critically low"

    def format_harvest_history(self, last_n: int = 8) -> str:
        """
        Format recent harvest history as bullet-point blocks, one per season.
        Shows each agent's harvest, community total, and stock status after.
        """
        history = self.state.harvest_history[-last_n:]
        if not history:
            return "No harvest history yet — this is the first season."

        blocks = []
        for entry in history:
            tick   = entry["tick"]
            actual = entry.get("actual", entry.get("harvests", {}))
            total  = entry["total"]
            stock_after = entry.get("stock_after", 0.0)
            status = self._status_from_stock(stock_after, self.state.max_stock)

            max_val = max(actual.values()) if actual else 0
            lines   = [f"Season {tick}:"]
            for aid, amt in sorted(actual.items()):
                marker = "  \u2190 highest" if amt == max_val and amt > 0 else ""
                lines.append(f"  {aid}: {amt:.1f} units{marker}")
            lines.append(f"  Community total: {total:.1f} units")
            lines.append(f"  Stock after: {status}")
            blocks.append("\n".join(lines))

        return "\n\n".join(blocks)

    def natural_language_state(self) -> str:
        """2-3 sentence natural language description for agent prompts."""
        desc = self._stock_description()
        trend = self._stock_trend()
        if trend == "unknown":
            trend_sentence = "This is an early season — trend data is limited."
        else:
            trend_sentence = f"The trend over recent seasons has been {trend}."
        return (
            f"{desc} "
            f"{trend_sentence} "
            f"There are {self.n_agents} fishers sharing this stock."
        )
