import argparse

import pandas as pd

from src import discover_and_label as dl


def _sample_returns() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=6, freq="B")
    return pd.DataFrame(
        {
            "AAPL": [0.01, -0.02, 0.005, 0.01, -0.003, 0.006],
            "MSFT": [0.008, -0.01, 0.006, 0.009, -0.002, 0.004],
        },
        index=dates,
    )


def test_parse_accepts_residualization_args(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "discover_and_label.py",
            "--symbols",
            "SPY",
            "--method",
            "PCA",
            "--residualization-method",
            "kalman",
            "--residualization-window",
            "84",
            "--residualization-halflife",
            "42",
            "--residualization-min-observations",
            "25",
            "--residualization-kalman-process-variance",
            "0.0002",
            "--residualization-kalman-observation-variance",
            "0.0025",
            "--residualization-kalman-initial-covariance",
            "15.0",
        ],
    )

    args = dl._parse()

    assert args.residualization_method == "kalman"
    assert args.residualization_window == 84
    assert args.residualization_halflife == 42.0
    assert args.residualization_min_observations == 25
    assert args.residualization_kalman_process_variance == 0.0002
    assert args.residualization_kalman_observation_variance == 0.0025
    assert args.residualization_kalman_initial_covariance == 15.0


def test_fit_passes_residualization_kwargs_to_statistical(monkeypatch):
    captured = {}

    def fake_statistical_factors(*args, **kwargs):
        captured.update(kwargs)
        fac = pd.DataFrame({"F1": [0.1, 0.2]}, index=[0, 1])
        load = pd.DataFrame({"F1": [0.6, 0.4]}, index=["AAPL", "MSFT"])
        return fac, load

    monkeypatch.setattr(dl, "statistical_factors", fake_statistical_factors)

    args = argparse.Namespace(
        method="PCA",
        k=2,
        residualization_method="kalman",
        residualization_window=96,
        residualization_halflife=50.0,
        residualization_min_observations=35,
        residualization_kalman_process_variance=3e-4,
        residualization_kalman_observation_variance=0.0015,
        residualization_kalman_initial_covariance=20.0,
    )

    dl._fit(_sample_returns(), args, cache_backend="dummy-backend")

    assert captured["cache_backend"] == "dummy-backend"
    assert captured["residualization_method"] == "kalman"
    assert captured["residualization_window"] == 96
    assert captured["residualization_halflife"] == 50.0
    assert captured["residualization_min_observations"] == 35
    assert captured["residualization_kalman_process_variance"] == 3e-4
    assert captured["residualization_kalman_observation_variance"] == 0.0015
    assert captured["residualization_kalman_initial_covariance"] == 20.0


def test_fit_backward_compatible_defaults(monkeypatch):
    captured = {}

    def fake_statistical_factors(*args, **kwargs):
        captured.update(kwargs)
        fac = pd.DataFrame({"F1": [0.1, 0.2]}, index=[0, 1])
        load = pd.DataFrame({"F1": [0.6, 0.4]}, index=["AAPL", "MSFT"])
        return fac, load

    monkeypatch.setattr(dl, "statistical_factors", fake_statistical_factors)

    args = argparse.Namespace(method="PCA", k=2)
    dl._fit(_sample_returns(), args, cache_backend=None)

    assert captured["residualization_method"] == "rolling"
    assert captured["residualization_window"] == 126
    assert captured["residualization_halflife"] == 63.0
    assert captured["residualization_min_observations"] == 30
    assert captured["residualization_kalman_process_variance"] == 1e-5
    assert captured["residualization_kalman_observation_variance"] is None
    assert captured["residualization_kalman_initial_covariance"] == 10.0


def test_fit_passes_residualization_kwargs_to_autoencoder(monkeypatch):
    captured = {}

    def fake_autoencoder_factors(*args, **kwargs):
        captured.update(kwargs)
        fac = pd.DataFrame({"F1": [0.1, 0.2]}, index=[0, 1])
        load = pd.DataFrame({"F1": [0.6, 0.4]}, index=["AAPL", "MSFT"])
        return fac, load

    monkeypatch.setattr(dl, "autoencoder_factors", fake_autoencoder_factors)

    args = argparse.Namespace(
        method="AE",
        k=3,
        residualization_method="ewm",
        residualization_window=120,
        residualization_halflife=40.0,
        residualization_min_observations=28,
        residualization_kalman_process_variance=1e-4,
        residualization_kalman_observation_variance=None,
        residualization_kalman_initial_covariance=12.0,
    )

    dl._fit(_sample_returns(), args, cache_backend="dummy-backend")

    assert captured["k"] == 3
    assert captured["cache_backend"] == "dummy-backend"
    assert captured["residualization_method"] == "ewm"
    assert captured["residualization_window"] == 120
    assert captured["residualization_halflife"] == 40.0
    assert captured["residualization_min_observations"] == 28
    assert captured["residualization_kalman_process_variance"] == 1e-4
    assert captured["residualization_kalman_observation_variance"] is None
    assert captured["residualization_kalman_initial_covariance"] == 12.0
