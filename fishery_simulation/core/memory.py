"""
Park et al. (2023) memory stream. All agent state lives here.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MemoryObject:
    content: str
    created_tick: int
    importance: float
    memory_type: str
    last_accessed_tick: int = field(init=False)

    def __post_init__(self):
        self.last_accessed_tick = self.created_tick


class MemoryStream:
    def __init__(self):
        self.memories: list = []
        self._importance_since_reflection: float = 0.0
        self._reflection_threshold: float = 100.0

    def add(
        self,
        content: str,
        importance: float,
        memory_type: str,
        tick: int,
    ) -> MemoryObject:
        m = MemoryObject(
            content=content,
            created_tick=tick,
            importance=importance,
            memory_type=memory_type,
        )
        self.memories.append(m)
        self._importance_since_reflection += importance
        return m

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        current_tick: int = 0,
        type_filter: list = None,
    ) -> list:
        """
        Score = (recency + importance_norm + relevance) / 3
        recency = 0.995 ^ (current_tick - last_accessed_tick)
        importance_norm = memory.importance / 10.0
        relevance = 0.5 (fixed — no embeddings)
        """
        candidates = self.memories
        if type_filter:
            candidates = [m for m in candidates if m.memory_type in type_filter]

        def score(m: MemoryObject) -> float:
            recency = 0.995 ** max(0, current_tick - m.last_accessed_tick)
            importance_norm = m.importance / 10.0
            relevance = 0.5
            return (recency + importance_norm + relevance) / 3.0

        scored = sorted(candidates, key=score, reverse=True)
        top = scored[:top_k]

        # Update last accessed
        for m in top:
            m.last_accessed_tick = current_tick

        return top

    def should_reflect(self, threshold: float = 100.0) -> bool:
        """Returns True when accumulated importance exceeds threshold."""
        if self._importance_since_reflection >= threshold:
            self._importance_since_reflection = 0.0
            return True
        return False

    def get_recent(self, n: int = 20) -> list:
        """Return n most recent regardless of score."""
        return self.memories[-n:]

    def format_for_prompt(self, memories: list) -> str:
        """Format retrieved memories as bullet list grouped by type."""
        if not memories:
            return "  (no relevant memories yet)"

        groups = {
            "harvest_decision": [],
            "observation": [],
            "dialogue": [],
            "reflection": [],
            "norm_belief": [],
        }

        for m in memories:
            t = m.memory_type
            if t in groups:
                groups[t].append(m)
            else:
                groups.setdefault("other", []).append(m)

        lines = []

        if groups["harvest_decision"]:
            lines.append("Past decisions:")
            for m in groups["harvest_decision"]:
                lines.append(f"  - [tick {m.created_tick}] {m.content}")

        if groups["observation"]:
            lines.append("What others have done:")
            for m in groups["observation"]:
                lines.append(f"  - [tick {m.created_tick}] {m.content}")

        if groups["dialogue"]:
            lines.append("Conversations:")
            for m in groups["dialogue"]:
                lines.append(f"  - [tick {m.created_tick}] {m.content}")

        if groups["norm_belief"]:
            lines.append("Norm beliefs:")
            for m in groups["norm_belief"]:
                lines.append(f"  - [tick {m.created_tick}] {m.content}")

        if groups["reflection"]:
            lines.append("Reflections:")
            for m in groups["reflection"]:
                lines.append(f"  - [tick {m.created_tick}] {m.content}")

        return "\n".join(lines) if lines else "  (no relevant memories yet)"
