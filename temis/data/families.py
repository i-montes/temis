"""Where the model is weakest, so data generation and the review queue can focus there.

Weights come from the validation agreement with Jev on 2026-11-08 (v1108-1022): tool_choice 61 %,
memory_op 62 %, memory_relation 69 %, note_link 71 % ... note_tags 100 %. A weight of 3 means the family
is drawn three times as often as one with weight 1. Re-measure and adjust as the model improves.
"""

from __future__ import annotations

FAMILY_WEIGHT = {
    "memory_op": 3.0, "tool_choice": 3.0, "memory_relation": 2.5, "note_link": 2.0,
    "injection": 1.5, "needs_memory": 1.5, "worth_remembering": 1.2, "asks_to_forget": 1.0,
    "tools": 0.8, "command_safety": 0.8, "go_no_go": 0.5, "note_tags": 0.4,
}
WEAK = {k for k, w in FAMILY_WEIGHT.items() if w >= 2.0}
# LLM-designed schemas have free workflow names: these words point to a weak family
WEAK_HINTS = ("memor", "relation", "relacion", "link", "enlace", "enlazar", "tool", "herramienta")


def queue_bonus(workflow: str | None) -> float:
    """Extra priority in the review queue for cases from weak families."""
    w = (workflow or "").lower()
    if w in WEAK:
        return 0.3
    if any(h in w for h in WEAK_HINTS):
        return 0.15
    return 0.0

# Families of the hard cases written by another model (docs/prompt_casos_dificiles.md,
# docs/prompt_rubrica_corta.md): the task formats JevBench's hard tier measures, in our own scenarios.
HARD_FAMILIES = ("long_policy", "multi_hop", "trap", "ambiguous", "adversarial", "tradeoff", "routing_hard",
                 "judge_hard", "temporal_numeric", "probability", "state_tracking", "rubric_short")
