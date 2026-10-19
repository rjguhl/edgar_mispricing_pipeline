"""How well does Polymarket price quarterly earnings beats?

Uses the resolved contracts assembled by src.modeling.backtest (run that
first) and their cached hourly price histories.

1. Calibration regression: logit P(beat) = a + b * logit(price). A
   calibrated market has a = 0, b = 1. a > 0 means beats are underpriced
   across the board; b < 1 means prices are too extreme (overconfident).
2. Slices: mean (outcome - price) by price bucket, EPS basis, loss vs.
   profit consensus, volume tercile and quarter, with Benjamini-Hochberg
   correction because many slices are tested at once.
3. Price dynamics: accuracy 5 / 3 / 1 days before the release, and whether
   the drift into the release predicts the outcome beyond the final price
   (under- or over-reaction to late information).

Run:
    python -m src.research.beat_pricing
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.modeling.backtest import PRICE_CACHE_DIR, RESULTS_PATH

logger = logging.getLogger(__name__)

HORIZONS_DAYS = (5, 3, 1)
PROB_CLIP = (0.01, 0.99)
FDR_ALPHA = 0.10


def logit(p: pd.Series) -> pd.Series:
    p = p.clip(*PROB_CLIP)
    return np.log(p / (1 - p))


def load() -> pd.DataFrame:
    """Backtest results plus the YES price at each horizon before release."""
    df = pd.read_parquet(RESULTS_PATH)
    for h in HORIZONS_DAYS:
        df[f"price_{h}d"] = np.nan
    for i, row in df.iterrows():
        path = PRICE_CACHE_DIR / f"{row['contract_id']}.json"
        if not path.exists():
            continue
        history = json.loads(path.read_text())
        release_ts = row["release_date"].tz_localize("UTC").timestamp()
        for h in HORIZONS_DAYS:
            before = [pt["p"] for pt in history if pt["t"] <= release_ts - h * 86400]
            if before:
                df.at[i, f"price_{h}d"] = float(before[-1])
    df["residual"] = df["outcome"] - df["market_prob"]
    return df


# ----------------------------------------------------------------------
# 1. Calibration regression
# ----------------------------------------------------------------------


def calibration_regression(df: pd.DataFrame, price_col: str = "market_prob") -> dict:
    X = sm.add_constant(logit(df[price_col]).rename("logit_price"))
    fit = sm.Logit(df["outcome"], X).fit(disp=0)
    wald = fit.wald_test("const = 0, logit_price = 1", scalar=True)
    ci = fit.conf_int()
    return {
        "n": int(fit.nobs),
        "intercept": fit.params["const"],
        "intercept_ci": tuple(ci.loc["const"]),
        "intercept_p": fit.pvalues["const"],
        "slope": fit.params["logit_price"],
        "slope_ci": tuple(ci.loc["logit_price"]),
        "slope_p_vs_1": 2 * stats.norm.sf(abs((fit.params["logit_price"] - 1) / fit.bse["logit_price"])),
        "joint_p": float(wald.pvalue),
    }


# ----------------------------------------------------------------------
# 2. Slices
# ----------------------------------------------------------------------


def slice_table(df: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "price": pd.cut(df["market_prob"], [0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True).astype(str),
        "eps_basis": df["eps_basis"].replace("", "unknown"),
        "consensus": np.where(df["eps_threshold"] < 0, "loss", "profit"),
        "volume": pd.qcut(df["volume"], 3, labels=["low", "mid", "high"]).astype(str),
        "quarter": df["release_date"].dt.to_period("Q").astype(str),
    }
    rows = []
    for dim, labels in groups.items():
        for label, idx in df.groupby(labels).groups.items():
            r = df.loc[idx, "residual"]
            t = stats.ttest_1samp(r, 0.0)
            half = stats.t.ppf(0.975, len(r) - 1) * r.std(ddof=1) / np.sqrt(len(r))
            rows.append(
                {
                    "slice": f"{dim}={label}",
                    "n": len(r),
                    "avg_price": df.loc[idx, "market_prob"].mean(),
                    "beat_rate": df.loc[idx, "outcome"].mean(),
                    "edge_cents": 100 * r.mean(),
                    "ci_low": 100 * (r.mean() - half),
                    "ci_high": 100 * (r.mean() + half),
                    "p": t.pvalue,
                }
            )
    out = pd.DataFrame(rows)
    out["q_bh"] = multipletests(out["p"], method="fdr_bh")[1]
    out["significant"] = out["q_bh"] < FDR_ALPHA
    return out


# ----------------------------------------------------------------------
# 3. Price dynamics
# ----------------------------------------------------------------------


def horizon_table(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy at each horizon on the contracts priced at every horizon."""
    cols = [f"price_{h}d" for h in HORIZONS_DAYS] + ["market_prob"]
    sub = df.dropna(subset=cols)
    rows = []
    for col in cols:
        p = sub[col].clip(*PROB_CLIP)
        rows.append(
            {
                "horizon": col.replace("price_", "").replace("market_prob", "release eve"),
                "n": len(sub),
                "brier": ((sub["outcome"] - p) ** 2).mean(),
                "roc_auc": _auc(sub["outcome"], p),
                "avg_price": p.mean(),
            }
        )
    return pd.DataFrame(rows)


def drift_regression(df: pd.DataFrame, start_col: str = "price_3d") -> dict:
    """logit P(beat) = a + b * logit(final) + c * (logit(final) - logit(start)).

    c > 0: prices under-react (late moves continue in the same direction);
    c < 0: they over-react; c = 0: the final price already contains the drift.
    """
    sub = df.dropna(subset=[start_col])
    final = logit(sub["market_prob"]).rename("logit_final")
    drift = (final - logit(sub[start_col])).rename("drift")
    fit = sm.Logit(sub["outcome"], sm.add_constant(pd.concat([final, drift], axis=1))).fit(disp=0)
    return {
        "n": int(fit.nobs),
        "drift_coef": fit.params["drift"],
        "drift_ci": tuple(fit.conf_int().loc["drift"]),
        "drift_p": fit.pvalues["drift"],
        "share_moved_5pts": float((abs(sub["market_prob"] - sub[start_col]) >= 0.05).mean()),
        "moves_toward_outcome": float(
            (np.sign(sub["market_prob"] - sub[start_col]) == np.where(sub["outcome"] == 1, 1, -1))[
                abs(sub["market_prob"] - sub[start_col]) >= 0.05
            ].mean()
        ),
    }


def _auc(y: pd.Series, p: pd.Series) -> float:
    from sklearn.metrics import roc_auc_score

    return roc_auc_score(y, p)


def main() -> None:
    df = load()
    pd.set_option("display.width", 160)

    cal = calibration_regression(df)
    print(f"\n1. Calibration regression (n={cal['n']}): logit P(beat) = a + b * logit(price)")
    print(f"   a = {cal['intercept']:+.3f}  95% CI [{cal['intercept_ci'][0]:+.3f}, {cal['intercept_ci'][1]:+.3f}]  p = {cal['intercept_p']:.3f}")
    print(f"   b = {cal['slope']:.3f}  95% CI [{cal['slope_ci'][0]:.3f}, {cal['slope_ci'][1]:.3f}]  p(b = 1) = {cal['slope_p_vs_1']:.3f}")
    print(f"   joint test a = 0, b = 1: p = {cal['joint_p']:.3f}")

    print(f"\n2. Edge by slice (outcome - price, cents; BH-adjusted q < {FDR_ALPHA} marked):")
    print(slice_table(df).round({"avg_price": 3, "beat_rate": 3, "edge_cents": 1, "ci_low": 1, "ci_high": 1, "p": 3, "q_bh": 3}).to_string(index=False))

    print("\n3a. Accuracy by horizon before release:")
    print(horizon_table(df).round(4).to_string(index=False))
    for h in HORIZONS_DAYS:
        print(f"    coverage {h}d: {df[f'price_{h}d'].notna().mean():.0%}")

    d = drift_regression(df)
    print(f"\n3b. Drift from 3 days out to release eve (n={d['n']}):")
    print(f"    drift coefficient c = {d['drift_coef']:+.3f}  95% CI [{d['drift_ci'][0]:+.3f}, {d['drift_ci'][1]:+.3f}]  p = {d['drift_p']:.3f}")
    print(f"    {d['share_moved_5pts']:.0%} of contracts moved >= 5 pts; {d['moves_toward_outcome']:.0%} of those moved toward the eventual outcome")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
