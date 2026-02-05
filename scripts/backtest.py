import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

def get_historical_data(
    api_key: str,
    equity_symbol: str = 'SPY',
    bond_symbol:   str = 'IEF',
    start_date:    str = '1997-09-10'
) -> pd.DataFrame:
    """
    Fetches full-history adjusted closes from Alpha Vantage for SPY and IEF,
    computes daily simple returns, and filters from start_date onward.
    """
    data = {}
    for symbol in (equity_symbol, bond_symbol):
        print(f"Fetching {symbol}...")
        url = (
            'https://www.alphavantage.co/query'
            f'?function=TIME_SERIES_DAILY_ADJUSTED'
            f'&symbol={symbol}'
            f'&outputsize=full'
            f'&apikey={api_key}'
        )
        r = requests.get(url)
        r.raise_for_status()
        js = r.json()
        if "Time Series (Daily)" not in js:
            raise ValueError(f"No data for {symbol}: {js.get('Note') or js}")
        df = (
            pd.DataFrame.from_dict(js["Time Series (Daily)"], orient="index")
              .rename(columns={"5. adjusted close": symbol})
              .astype(float)
        )
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        data[symbol] = df[symbol]

    df = pd.DataFrame({
        "equity_returns": data[equity_symbol].pct_change(),
        "bond_returns":   data[bond_symbol].pct_change()
    }).dropna()

    return df.loc[df.index >= pd.to_datetime(start_date)]


def calculate_rebalancing_signals(
    market_data: pd.DataFrame,
    equity_target: float = 0.6
) -> pd.DataFrame:
    """
    Computes:
      • calendar_signal: deviation from monthly rebalanced weight
      • threshold_signal: average deviation over δ ∈ {0,…,2.5%}
    """
    signals = market_data.copy()
    signals["calendar_signal"] = 0.0

    # — Calendar signal —
    w = equity_target
    for t in range(1, len(signals)):
        re = signals["equity_returns"].iat[t - 1]
        rb = signals["bond_returns"].iat[t - 1]
        w = w * (1 + re) / (w * (1 + re) + (1 - w) * (1 + rb))
        signals.iat[t, signals.columns.get_loc("calendar_signal")] = w - equity_target

        # reset on first trading day of new month
        if signals.index[t].month != signals.index[t - 1].month:
            w = equity_target

    # — Threshold signal —
    deltas = np.arange(0.0, 0.0251, 0.001)
    devs = pd.DataFrame(index=signals.index)
    for δ in deltas:
        col = f"dev_{δ:.3f}"
        devs[col] = 0.0
        w = equity_target
        for t in range(1, len(signals)):
            re = signals["equity_returns"].iat[t - 1]
            rb = signals["bond_returns"].iat[t - 1]
            w = w * (1 + re) / (w * (1 + re) + (1 - w) * (1 + rb))
            d = w - equity_target
            devs.iat[t, devs.columns.get_loc(col)] = d
            if abs(d) > δ:
                w = equity_target  # immediate reversion

    signals["threshold_signal"] = devs.mean(axis=1)
    return signals


def generate_actionable_signal(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the front‑running weight:
      1) Threshold part: −threshold_signal / 0.015
      2) Calendar part: 
         – contrarian last 5 biz days 
         – reversal on 1st biz day uses t−4 signal
      3) Average the two
    """
    strat = signals.copy()

    # 1) Threshold
    threshold_w = - strat["threshold_signal"] / 0.015

    # 2) Calendar
    # mark last 5 business days of month
    strat["is_last_week"] = [
        ((dt + pd.tseries.offsets.BMonthEnd(0)) - dt).days < 5
        for dt in strat.index
    ]

    cal_w = pd.Series(0.0, index=strat.index)
    mask = strat["is_last_week"]
    cal_w[mask] = -np.sign(strat.loc[mask, "calendar_signal"])

    # first trading day reversal uses t−4
    first_day = strat.index.to_period("M") != strat.index.to_period("M").shift(1)
    cal_w[first_day] = np.sign(strat["calendar_signal"].shift(4)[first_day])

    # 3) combine
    strat["actionable_signal"] = (threshold_w + cal_w) / 2.0
    return strat


def backtest_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the signal to (equity – bond) returns, computes performance,
    and plots log‑scale cumulative P&L vs. buy‑and‑hold SPY.
    """
    df = df.copy()
    df["xa_returns"]       = df["equity_returns"] - df["bond_returns"]
    df["strategy_returns"] = df["actionable_signal"].shift(1) * df["xa_returns"]
    df = df.dropna()

    # benchmark
    df["cum_equity"] = (1 + df["equity_returns"]).cumprod()

    days = len(df)
    tot_s = df["strategy_returns"].prod() - 1
    ann_s = (1 + tot_s) ** (252 / days) - 1
    vol_s = df["strategy_returns"].std() * np.sqrt(252)
    sr_s  = ann_s / vol_s

    tot_e = df["cum_equity"].iat[-1] - 1
    ann_e = (1 + tot_e) ** (252 / days) - 1
    vol_e = df["equity_returns"].std() * np.sqrt(252)
    sr_e  = ann_e / vol_e

    print(f"\nStrategy annualized return: {ann_s:.2%}")
    print(f"Strategy vol:                {vol_s:.2%}")
    print(f"Strategy Sharpe:             {sr_s:.2f}\n")

    print(f"Benchmark (SPY) annualized return: {ann_e:.2%}")
    print(f"Benchmark vol:                   {vol_e:.2%}")
    print(f"Benchmark Sharpe:               {sr_e:.2f}")

    # plot scaled P&L
    scale = vol_e / vol_s if vol_s else 1
    df["strat_scaled"] = df["strategy_returns"] * scale
    df["cum_strat"]    = (1 + df["strat_scaled"]).cumprod()

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["cum_strat"], label="Front‑Run Strategy (scaled)")
    ax.plot(df["cum_equity"], label="Buy & Hold SPY", linestyle="--")
    ax.set_yscale("log")
    ax.set_title("Strategy vs. SPY (Log Scale)")
    ax.legend()
    plt.tight_layout()
    plt.show()

    return df


if __name__ == "__main__":
    from src.config import config

    AV_API_KEY = config.ALPHAVANTAGE_API_KEY
    if not AV_API_KEY:
        raise SystemExit(
            "ALPHAVANTAGE_API_KEY is not set. Add it to your .env file (project root) "
            "or export it in your shell environment."
        )

    data    = get_historical_data(AV_API_KEY)
    signals = calculate_rebalancing_signals(data)
    strat   = generate_actionable_signal(signals)
    results = backtest_strategy(strat)

    print("\nLast 5 rows of signals & returns:")
    print(
        results[
            [
                "equity_returns","bond_returns",
                "threshold_signal","calendar_signal",
                "actionable_signal","strategy_returns"
            ]
        ].tail().round(4)
    )
