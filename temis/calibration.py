"""Temperatures for the student, fitted on held-out logits.

Three families live in config.json next to the model:
  temperatures            soft: match the teachers' distributions (best KL; what the KL benchmarks measure)
  temperatures_accuracy   accuracy: a 70 % answer is right ~70 % of the time (the API/UI default)
  temperatures_jevbench   by type and number of options, aimed at JevBench v1.5: for noul the temperature
                          that gets the most answers right AND outside its abstention band (0.20, 0.80);
                          for score the one that minimizes the expected level's normalized MAE;
                          for choice the NLL one. Groups with fewer than MIN_ROWS rows use the per-type value "*".
"""

from __future__ import annotations

import numpy as np

GRID = np.arange(0.2, 2.51, 0.05)
# noul/score get a higher floor: at 0.2 they turn maximally confident and the ECE doubles
GRID_DECISIVE = np.arange(0.4, 2.51, 0.05)
MIN_ROWS = 30
NOUL_BAND = (0.20, 0.80)

Rows = list[tuple[np.ndarray, int]]  # (logits over the options, index of the right option)


def softmax(z: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / t
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def fit_nll(rows: Rows, grid: np.ndarray = GRID) -> float:
    def nll(t: float) -> float:
        return -float(np.mean([np.log(max(softmax(lg, t)[y], 1e-9)) for lg, y in rows]))
    return float(min(grid, key=nll))


def fit_noul_band(rows: Rows, lo: float = NOUL_BAND[0], hi: float = NOUL_BAND[1], grid: np.ndarray = GRID_DECISIVE) -> float:
    """Temperature with the most answers that are right and decided (p_true <= lo or >= hi); ties -> closest to 1."""
    def decided_right(t: float) -> int:
        n = 0
        for lg, y in rows:
            p = softmax(lg, t)
            p_true = p[1] if len(p) > 1 else p[0]
            n += int(p.argmax() == y and (p_true <= lo or p_true >= hi))
        return n
    return float(max(grid, key=lambda t: (decided_right(t), -abs(t - 1.0))))


def fit_score_nmae(rows: Rows, grid: np.ndarray = GRID_DECISIVE) -> float:
    def nmae(t: float) -> float:
        tot = 0.0
        for lg, y in rows:
            p = softmax(lg, t)
            k = len(p)
            tot += abs(float((np.arange(k) * p).sum()) - y) / max(k - 1, 1)
        return tot / len(rows)
    return float(min(grid, key=lambda t: (round(nmae(t), 6), abs(t - 1.0))))


FIT = {"choice": fit_nll, "noul": fit_noul_band, "score": fit_score_nmae}


def fit_jevbench(by_kind_n: dict[tuple[str, int], Rows], min_rows: int = MIN_ROWS) -> dict[str, dict[str, float]]:
    """{"choice": {"3": T, "4": T, "*": T}, "noul": {...}, "score": {...}}; "*" is fitted on all rows of the type."""
    out: dict[str, dict[str, float]] = {}
    by_kind: dict[str, Rows] = {}
    for (kind, n), rows in by_kind_n.items():
        by_kind.setdefault(kind, []).extend(rows)
        if len(rows) >= min_rows:
            out.setdefault(kind, {})[str(n)] = round(FIT[kind](rows), 2)
    for kind, rows in by_kind.items():
        if rows:
            out.setdefault(kind, {})["*"] = round(FIT[kind](rows), 2)
    return out


def pick_temperature(temps: dict, kind: str, n: int) -> float:
    t = temps.get(kind)
    if isinstance(t, dict):
        return float(t.get(str(n), t.get("*", 1.0)))
    if isinstance(t, (int, float)):
        return float(t)
    return 1.0


__all__ = ["softmax", "fit_nll", "fit_noul_band", "fit_score_nmae", "fit_jevbench", "pick_temperature", "GRID"]
