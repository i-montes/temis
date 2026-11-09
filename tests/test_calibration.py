import numpy as np

from temis.calibration import fit_jevbench, fit_noul_band, fit_score_nmae, pick_temperature


def test_fit_noul_band_sharpens_correct_hesitant_answers():
    rows = [(np.array([0.0, 0.6]), 1)] * 10 + [(np.array([0.6, 0.0]), 0)] * 10  # p_true ≈ 0.65 / 0.35: in the band
    t = fit_noul_band(rows)
    assert t < 0.5                      # at T=1 both sit in the band; sharpening takes them out without changing the argmax
    p = np.exp(rows[0][0] / t)
    assert p[1] / p.sum() >= 0.8


def test_fit_score_nmae_prefers_sharp_when_argmax_right():
    rows = [(np.array([0.0, 1.0, 0.0, 0.0]), 1)] * 5
    assert fit_score_nmae(rows) <= 1.0


def test_pick_temperature_falls_back():
    temps = {"choice": {"4": 0.7, "*": 0.9}}
    assert pick_temperature(temps, "choice", 4) == 0.7 and pick_temperature(temps, "choice", 7) == 0.9
    assert pick_temperature(temps, "noul", 2) == 1.0


def test_fit_jevbench_groups_small_n_under_star():
    rows = [(np.array([0.0, 0.6]), 1)] * 40
    temps = fit_jevbench({("noul", 2): rows, ("choice", 3): [(np.array([2.0, 0.0, 0.0]), 0)] * 5})
    assert "2" in temps["noul"] and "*" in temps["noul"]
    assert "3" not in temps["choice"] and "*" in temps["choice"]   # fewer than 30 rows: only the per-type value
