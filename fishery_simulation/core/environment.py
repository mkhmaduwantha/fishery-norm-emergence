"""
Shared fish stock using logistic growth dynamics.

Temporal framing: 4 seasons per year.
  tick 0  = Year 1, Season 1
  tick 3  = Year 1, Season 4
  tick 4  = Year 2, Season 1
  etc.

Collapse and recovery:
  When the stock collapses after a harvest the remaining seasons of that
  year are filled with synthetic "fishery closed" history entries and
  the tick jumps to the first season of the NEXT year with stock fully
  reset to initial level.  Agents never experience a closed season as
  an active simulation tick — they see the closure only in history.
"""
from dataclasses import dataclass, field


SEASONS_PER_YEAR           = 4
DEFAULT_INITIAL_STOCK      = 80.0
DEFAULT_MAX_STOCK          = 100.0
DEFAULT_REGROWTH_RATE      = 0.3
DEFAULT_COLLAPSE_THRESHOLD = 20.0
MAX_INDIVIDUAL_HARVEST     = 10.0
SUSTAINABLE_TOTAL          = 12.0   # reference value, not enforced


@dataclass
class FisheryState:
    stock: float
    max_stock: float
    regrowth_rate: float
    collapse_threshold: float
    initial_stock: float
    collapsed: bool = False          # never True during active simulation ticks
    tick: int = 0
    collapse_count: int = 0
    harvest_history: list = field(default_factory=list)
    # Each entry: {tick, year, season_in_year, harvests, actual,
    #              total, stock_after, collapsed, closed_season}


class FisheryEnvironment:
    def __init__(
        self,
        n_agents: int = 8,
        initial_stock: float = DEFAULT_INITIAL_STOCK,
        max_stock: float = DEFAULT_MAX_STOCK,
        regrowth_rate: float = DEFAULT_REGROWTH_RATE,
        collapse_threshold: float = DEFAULT_COLLAPSE_THRESHOLD,
    ):
        self.n_agents       = n_agents
        self._initial_stock = initial_stock
        self.state = FisheryState(
            stock=initial_stock,
            max_stock=max_stock,
            regrowth_rate=regrowth_rate,
            collapse_threshold=collapse_threshold,
            initial_stock=initial_stock,
        )

    # ── Temporal helpers ──────────────────────────────────────────

    @staticmethod
    def tick_to_year_season(tick: int) -> tuple:
        """Return (year_1based, season_1based) for a given tick."""
        year   = tick // SEASONS_PER_YEAR + 1
        season = tick  % SEASONS_PER_YEAR + 1
        return year, season

    @staticmethod
    def season_label(tick: int) -> str:
        """'Year 2, Season 3' style label."""
        year, season = FisheryEnvironment.tick_to_year_season(tick)
        return f"Year {year}, Season {season}"

    # ── Core simulation ───────────────────────────────────────────

    def apply_harvests(self, harvests: dict) -> dict:
        """
        Apply all agents' harvests simultaneously.

        On collapse:
          1. Record the collapse season in history.
          2. Add synthetic "fishery closed" entries for the remaining
             seasons of the current year.
          3. Advance tick to the first season of the next year.
          4. Reset stock to initial_stock.
          Agents never experience a closed tick — only see it in history.

        Returns dict of actual yields.
        """
        s    = self.state
        tick = s.tick
        year, season_num = self.tick_to_year_season(tick)

        # ── Normal season ─────────────────────────────────────────
        stock_before    = s.stock          # capture before any deduction
        total_requested = sum(harvests.values())
        available       = s.stock

        if total_requested > available and total_requested > 0:
            scale  = available / total_requested
            actual = {aid: h * scale for aid, h in harvests.items()}
        else:
            actual = dict(harvests)

        total_actual = sum(actual.values())
        s.stock      = max(0.0, s.stock - total_actual)

        # Check collapse BEFORE regrowth
        if s.stock < s.collapse_threshold:
            s.collapse_count += 1

            # Record the collapse season
            s.harvest_history.append({
                "tick":           tick,
                "year":           year,
                "season_in_year": season_num,
                "harvests":       dict(harvests),
                "actual":         actual,
                "total":          total_actual,
                "stock_before":   stock_before,
                "stock_after":    0.0,
                "collapsed":      True,
                "closed_season":  False,
            })
            s.tick += 1

            # Synthetic closed-season entries for remaining seasons this year
            seasons_remaining = SEASONS_PER_YEAR - season_num
            for i in range(seasons_remaining):
                closed_tick = s.tick
                c_year, c_season = self.tick_to_year_season(closed_tick)
                zero_row = {aid: 0.0 for aid in harvests}
                s.harvest_history.append({
                    "tick":           closed_tick,
                    "year":           c_year,
                    "season_in_year": c_season,
                    "harvests":       zero_row,
                    "actual":         dict(zero_row),
                    "total":          0.0,
                    "stock_after":    0.0,
                    "collapsed":      True,
                    "closed_season":  True,
                })
                s.tick += 1

            # Jump to first season of next year, reset stock
            # (tick is already at the right place after the loop above)
            s.stock    = s.initial_stock
            s.collapsed = False   # stays False — no closed ticks will run

            return actual

        # ── Logistic regrowth (normal path) ───────────────────────
        growth  = s.regrowth_rate * s.stock * (1 - s.stock / s.max_stock)
        s.stock = min(s.max_stock, s.stock + growth)

        s.harvest_history.append({
            "tick":           tick,
            "year":           year,
            "season_in_year": season_num,
            "harvests":       dict(harvests),
            "actual":         actual,
            "total":          total_actual,
            "stock_before":   stock_before,
            "stock_after":    s.stock,
            "collapsed":      False,
            "closed_season":  False,
        })
        s.tick += 1
        return actual

    # ── Observation / description helpers ─────────────────────────

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
        normal = [
            h for h in self.state.harvest_history
            if not h.get("closed_season", False) and not h.get("collapsed", False)
        ]
        if len(normal) < 3:
            return "unknown"
        recent = normal[-3:]
        stocks = [h["stock_after"] for h in recent]
        delta  = stocks[-1] - stocks[0]
        if delta < -5:
            return "declining"
        elif delta > 3:
            return "recovering"
        else:
            return "stable"

    def get_observation_for(self, agent_id: str, last_harvests: dict) -> dict:
        s   = self.state
        pct = s.stock / s.max_stock if s.max_stock > 0 else 0.0

        others_harvests = {
            aid: amt for aid, amt in last_harvests.items()
            if aid != agent_id
        }
        my_last = last_harvests.get(agent_id)
        year, season_num = self.tick_to_year_season(s.tick)

        return {
            "stock_description": self._stock_description(),
            "stock_trend":       self._stock_trend(),
            "stock":             s.stock,
            "max_stock":         s.max_stock,
            "stock_pct":         pct,
            "others_harvests":   others_harvests,
            "my_last_harvest":   my_last,
            "tick":              s.tick,
            "year":              year,
            "season_in_year":    season_num,
            "season_label":      self.season_label(s.tick),
            "n_agents":          self.n_agents,
            "collapsed":         False,   # always False — no active closed ticks
        }

    @staticmethod
    def _status_from_stock(stock_after: float, max_stock: float,
                            closed: bool = False) -> str:
        if closed:
            return "fishery closed (recovering)"
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
            return "critically low / collapsed"

    def format_harvest_history(self, last_n: int = 5) -> str:
        """
        Format the last N normal seasons (skipping synthetic closed rows)
        plus any collapse/closed entries that immediately precede them,
        so agents see both the collapse event and the recovery context.
        """
        history = self.state.harvest_history
        if not history:
            return "No harvest history yet — this is the first season."

        # Take last_n non-closed-season entries, then include any
        # immediately preceding collapse/closed rows for context
        normal = [h for h in history if not h.get("closed_season", False)]
        recent_normal = normal[-last_n:]
        if not recent_normal:
            return "No harvest history yet — this is the first season."

        first_tick = recent_normal[0]["tick"]

        # Include all history from first_tick onwards (catches closed rows too)
        window = [h for h in history if h["tick"] >= first_tick]

        blocks    = []
        prev_year = None

        max_stock = self.state.max_stock

        for entry in window:
            year        = entry.get("year",  self.tick_to_year_season(entry["tick"])[0])
            season_num  = entry.get("season_in_year",
                                     self.tick_to_year_season(entry["tick"])[1])
            closed      = entry.get("closed_season",  False)
            collapsed_e = entry.get("collapsed",       False)
            actual      = entry.get("actual", entry.get("harvests", {}))
            total       = entry["total"]
            stock_before = entry.get("stock_before", None)
            stock_after  = entry.get("stock_after", 0.0)

            # New-year divider — show the recovered stock level
            if prev_year is not None and year != prev_year:
                recovered = self.state.initial_stock
                blocks.append(
                    f"─── Year {year} begins — stock recovered to "
                    f"{recovered:.1f} / {max_stock:.1f} units ───"
                )
            prev_year = year

            label = f"Year {year}, Season {season_num}:"

            if closed:
                blocks.append(
                    f"{label}\n"
                    f"  [FISHERY CLOSED — no fishing, stock recovering]"
                )
            elif collapsed_e:
                max_val = max(actual.values()) if actual else 0
                lines   = [f"{label}  ⚠ Stock collapsed after this season"]
                if stock_before is not None:
                    lines.append(f"  Stock at start: {stock_before:.1f} / {max_stock:.1f} units")
                for aid, amt in sorted(actual.items()):
                    marker = "  ← highest" if amt == max_val and amt > 0 else ""
                    lines.append(f"  {aid}: {amt:.1f} units{marker}")
                lines.append(f"  Community total: {total:.1f} units")
                lines.append(f"  Stock after harvest: collapsed (below {self.state.collapse_threshold:.1f} units)")
                blocks.append("\n".join(lines))
            else:
                status  = self._status_from_stock(stock_after, max_stock)
                max_val = max(actual.values()) if actual else 0
                lines   = [label]
                if stock_before is not None:
                    lines.append(f"  Stock at start: {stock_before:.1f} / {max_stock:.1f} units")
                for aid, amt in sorted(actual.items()):
                    marker = "  ← highest" if amt == max_val and amt > 0 else ""
                    lines.append(f"  {aid}: {amt:.1f} units{marker}")
                lines.append(f"  Community total: {total:.1f} units")
                lines.append(f"  Stock after season: {stock_after:.1f} / {max_stock:.1f} units ({status})")
                blocks.append("\n".join(lines))

        return "\n\n".join(blocks)

    def natural_language_state(self) -> str:
        """Natural language situation description for agent prompts."""
        s            = self.state
        year, season = self.tick_to_year_season(s.tick)
        desc         = self._stock_description()
        trend        = self._stock_trend()

        if trend == "unknown":
            trend_sentence = "This is an early season — trend data is limited."
        else:
            trend_sentence = f"The trend over recent seasons has been {trend}."

        return (
            f"It is Year {year}, Season {season}. "
            f"Available stock at the start of this season: {s.stock:.1f} / {s.max_stock:.1f} units. "
            f"{desc} "
            f"{trend_sentence} "
            f"There are {self.n_agents} fishers sharing this stock."
        )
