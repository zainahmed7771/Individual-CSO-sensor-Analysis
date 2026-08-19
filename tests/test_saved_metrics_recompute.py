from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def rmse(observed, predicted):
    return float(np.sqrt(np.mean((np.asarray(observed) - np.asarray(predicted)) ** 2)))


def test_original_locked_test_metrics_recompute():
    summary = pd.read_csv(ROOT / "07_machine_learning/results/original_selected_sensor/four_model_summary.csv").set_index("target")
    for target in ("beta", "scale", "duration", "frequency"):
        filename = "lambda" if target == "scale" else target
        pred = pd.read_csv(ROOT / f"07_machine_learning/results/original_selected_sensor/{filename}_test_predictions.csv")
        assert np.isclose(rmse(pred.observed_log, pred.predicted_log), summary.loc[target, "rmse_model_scale"], rtol=0, atol=1e-10)
        assert np.isclose(rmse(pred.observed_log, pred.dummy_predicted_log), summary.loc[target, "dummy_rmse_model_scale"], rtol=0, atol=1e-10)


def test_regional_locked_test_metrics_recompute():
    summary = pd.read_csv(ROOT / "07_machine_learning/results/regional_cluster/four_model_summary.csv").set_index("target")
    for target in ("beta", "lambda", "duration", "frequency"):
        pred = pd.read_csv(ROOT / f"07_machine_learning/results/regional_cluster/{target}_test_predictions.csv")
        assert np.isclose(rmse(pred.observed_transformed, pred.predicted_transformed), summary.loc[target, "rmse"], rtol=0, atol=1e-10)
        assert np.isclose(rmse(pred.observed_transformed, pred.dummy_predicted_transformed), summary.loc[target, "dummy_rmse"], rtol=0, atol=1e-10)
