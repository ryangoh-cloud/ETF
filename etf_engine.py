"""
ETF Regime Trading Engine
─────────────────────────
Gaussian HMM regime detection (5-state → bullish/neutral/bearish),
technical signal computation, weight optimization (max Sharpe),
backtesting, and trade blotter generation.
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn import hmm
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ETF_UNIVERSE = ["QQQ", "SOXL", "SOXX", "DFEN", "QTUM", "DTCR", "TLT"]

REGIME_COLORS = {
    "bullish": "rgba(0, 200, 100, 0.13)",
    "bearish": "rgba(255, 80, 80, 0.13)",
    "neutral":  "rgba(210, 153, 34, 0.06)",
}

REGIME_PALETTE = {
    "bullish": "#3fb950",
    "neutral":  "#d29922",
    "bearish": "#f85149",
}

# ──────────────────────────────────────────────────────────────────────────────
# Data Fetching
# ──────────────────────────────────────────────────────────────────────────────

def fetch_data(ticker: str, period: str = "3y") -> pd.DataFrame:
    """Download OHLCV via yfinance Ticker.history (robust to yf API changes)."""
    t = yf.Ticker(ticker)
    df = t.history(period=period, auto_adjust=True)
    # Strip timezone so all indices are tz-naive
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.columns = [c.lower() for c in df.columns]
    needed = ["open", "high", "low", "close", "volume"]
    df = df[[c for c in needed if c in df.columns]].dropna()
    return df


def validate_ticker(ticker: str) -> tuple[bool, str]:
    """
    Quick pre-flight check: does the ticker exist and return recent price data?
    Uses a 5-day history pull — minimal data transfer, fast response.

    Returns (is_valid, error_message).  error_message is "" on success.
    """
    sym = ticker.strip().upper()
    if not sym:
        return False, "Ticker symbol cannot be empty."
    try:
        t  = yf.Ticker(sym)
        df = t.history(period="5d")
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        if df.empty or len(df) < 2:
            return False, (
                f"**{sym}** returned no price data. "
                "The symbol may be delisted, misspelled, or not traded on a US exchange."
            )
        return True, ""
    except Exception as exc:
        return False, f"Network or API error while validating **{sym}**: {exc}"


# ──────────────────────────────────────────────────────────────────────────────
# HMM Features
# ──────────────────────────────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Three HMM features:
      - returns       : daily log-return proxy (pct_change)
      - norm_range    : (high - low) / close  — intraday volatility
      - vol_volatility: rolling std of volume pct-change  — volume regime
    """
    d = df.copy()
    d["returns"]     = d["close"].pct_change()
    d["norm_range"]  = (d["high"] - d["low"]) / d["close"]
    d["vol_vol"]     = d["volume"].pct_change().rolling(5, min_periods=3).std()
    features = (
        d[["returns", "norm_range", "vol_vol"]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    return features


# ──────────────────────────────────────────────────────────────────────────────
# HMM Regime Detection
# ──────────────────────────────────────────────────────────────────────────────

def fit_hmm(
    features: pd.DataFrame,
    n_components: int = 5,
) -> tuple[hmm.GaussianHMM, pd.Series, dict, StandardScaler]:
    """
    Fit a GaussianHMM with *n_components* states on standardised features.
    States are ranked by mean return and mapped:
      rank 0-1  → bullish
      rank 2    → neutral
      rank 3-4  → bearish
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    model = hmm.GaussianHMM(
        n_components=n_components,
        covariance_type="diag",
        n_iter=100,
        tol=1e-4,
        random_state=42,
        init_params="stmc",
        params="stmc",
    )
    model.fit(X)
    states = model.predict(X)

    # Rank states by their mean *unscaled* daily return
    raw_returns = features["returns"].values
    mean_ret_by_state = {
        s: raw_returns[states == s].mean() if (states == s).any() else 0.0
        for s in range(n_components)
    }
    ranked = sorted(mean_ret_by_state, key=mean_ret_by_state.get, reverse=True)

    regime_map: dict[int, str] = {}
    for rank, state in enumerate(ranked):
        if rank < 2:
            regime_map[state] = "bullish"
        elif rank == 2:
            regime_map[state] = "neutral"
        else:
            regime_map[state] = "bearish"

    regimes = pd.Series(
        [regime_map[s] for s in states],
        index=features.index,
        name="regime",
    )
    return model, regimes, regime_map, scaler


# ──────────────────────────────────────────────────────────────────────────────
# Technical Signals  (each returns a Series in roughly [-1, +1])
# ──────────────────────────────────────────────────────────────────────────────

def _zscore(series: pd.Series, window: int = 126, min_p: int = 30) -> pd.Series:
    mu  = series.rolling(window, min_periods=min_p).mean()
    sig = series.rolling(window, min_periods=min_p).std()
    return ((series - mu) / (sig + 1e-10)).clip(-3, 3) / 3


def signal_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """RSI normalised: oversold → +1 (buy), overbought → -1 (sell)."""
    delta = prices.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rsi   = 100 - 100 / (1 + gain / (loss + 1e-10))
    return ((50 - rsi) / 50).clip(-1, 1)


def signal_momentum(prices: pd.Series, period: int = 20) -> pd.Series:
    """Z-score of 20-day price momentum."""
    return _zscore(prices.pct_change(period))


def signal_volatility(prices: pd.Series, period: int = 20) -> pd.Series:
    """Inverse z-score of realised vol: low-vol environment → positive signal."""
    vol = prices.pct_change().rolling(period, min_periods=10).std() * np.sqrt(252)
    return -_zscore(vol)


def signal_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Directional ADX: sign(DI+ − DI−) × ADX/100."""
    high, low, close = df["high"], df["low"], df["close"]

    tr = pd.concat(
        [high - low,
         (high - close.shift(1)).abs(),
         (low  - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)

    up_move   = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm  = np.where((up_move   > down_move) & (up_move   > 0), up_move,   0.0)
    minus_dm = np.where((down_move > up_move)   & (down_move > 0), down_move, 0.0)
    plus_dm  = pd.Series(plus_dm,  index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    alpha = 1 / period
    atr      = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm( alpha=alpha, adjust=False).mean() / (atr + 1e-10)
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / (atr + 1e-10)

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    direction = np.sign(plus_di - minus_di)
    return (direction * (adx / 100).clip(0, 1)).rename("adx")


def signal_ema(prices: pd.Series, fast: int = 12, slow: int = 26, sig_p: int = 9) -> pd.Series:
    """MACD histogram z-score: positive when fast EMA leads slow EMA."""
    macd_line   = prices.ewm(span=fast, adjust=False).mean() - prices.ewm(span=slow, adjust=False).mean()
    signal_line = macd_line.ewm(span=sig_p, adjust=False).mean()
    hist        = (macd_line - signal_line) / (prices + 1e-10)
    return _zscore(hist)


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all five signals and return a clean DataFrame."""
    s = pd.DataFrame(index=df.index)
    s["rsi"]        = signal_rsi(df["close"])
    s["momentum"]   = signal_momentum(df["close"])
    s["volatility"] = signal_volatility(df["close"])
    s["adx"]        = signal_adx(df)
    s["ema"]        = signal_ema(df["close"])
    return s.replace([np.inf, -np.inf], np.nan).dropna()


# ──────────────────────────────────────────────────────────────────────────────
# Backtesting
# ──────────────────────────────────────────────────────────────────────────────

def _normalise_weights(w: np.ndarray) -> np.ndarray:
    w = np.abs(w)
    total = w.sum()
    return w / (total + 1e-10)


def backtest(
    df: pd.DataFrame,
    regimes: pd.Series,
    signals: pd.DataFrame,
    weights: np.ndarray,
    entry_threshold: float = 0.10,
) -> tuple[pd.Series, pd.Series]:
    """
    Long-only regime strategy.
      Entry : regime == bullish  AND  composite_score > entry_threshold
      Exit  : regime == bearish

    Uses raw numpy arrays inside the loop to avoid the ~50-100x overhead
    of pandas .iloc on every iteration — critical for optimisation speed.
    """
    idx = df.index.intersection(regimes.index).intersection(signals.index)
    if len(idx) < 10:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    n         = len(idx)
    reg_arr   = regimes.loc[idx].values           # object array (strings)
    sig_arr   = signals.loc[idx].values           # float64 (n, n_signals)
    close_arr = df.loc[idx, "close"].values       # float64 (n,)

    w        = _normalise_weights(weights)
    comp_arr = sig_arr @ w                        # composite score per day

    # State-machine loop — numpy indexing, no pandas overhead
    pos_arr  = np.zeros(n, dtype=np.float64)
    in_trade = False
    for i in range(1, n):
        if not in_trade:
            if reg_arr[i] == "bullish" and comp_arr[i] > entry_threshold:
                in_trade   = True
                pos_arr[i] = 1.0
        else:
            if reg_arr[i] == "bearish":
                in_trade = False          # pos_arr[i] stays 0.0
            else:
                pos_arr[i] = 1.0

    # Strategy return[i] = position[i-1] × pct_return[i]
    pct_ret  = np.diff(close_arr) / (close_arr[:-1] + 1e-10)   # length n-1
    strat_ret = pos_arr[:-1] * pct_ret                          # length n-1

    strategy_returns = pd.Series(strat_ret, index=idx[1:])
    position         = pd.Series(pos_arr,   index=idx)
    return strategy_returns, position


# ──────────────────────────────────────────────────────────────────────────────
# Performance Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(strategy_returns: pd.Series) -> dict:
    if len(strategy_returns) < 5:
        return dict(total_return=0.0, sharpe=0.0, win_rate=0.0, max_drawdown=0.0)

    total_return = float((1 + strategy_returns).prod() - 1)
    ann_ret      = float(strategy_returns.mean() * 252)
    ann_vol      = float(strategy_returns.std() * np.sqrt(252))
    sharpe       = ann_ret / (ann_vol + 1e-10)

    active       = strategy_returns[strategy_returns != 0]
    win_rate     = float((active > 0).mean()) if len(active) > 0 else 0.0

    cum          = (1 + strategy_returns).cumprod()
    max_dd       = float(((cum - cum.cummax()) / (cum.cummax() + 1e-10)).min())

    return dict(total_return=total_return, sharpe=sharpe, win_rate=win_rate, max_drawdown=max_dd)


# ──────────────────────────────────────────────────────────────────────────────
# Weight Optimisation  (maximise Sharpe over 1-year window)
# ──────────────────────────────────────────────────────────────────────────────

def optimize_weights(
    df: pd.DataFrame,
    regimes: pd.Series,
    signals: pd.DataFrame,
    n_random: int = 6,
) -> np.ndarray:
    """
    Nelder-Mead with structured + random restarts to maximise Sharpe.

    Starting points:
      - equal weight across all signals
      - one-hot for each individual signal  (n_signals points)
      - n_random random draws from Exponential
    Total starts = 1 + n_signals + n_random  (default: 12 for 5 signals)
    Each start is capped at 150 iterations — fast with the vectorised backtest.
    """
    n_sig        = signals.shape[1]
    best_sharpe  = -np.inf
    best_weights = np.ones(n_sig) / n_sig

    def objective(w: np.ndarray) -> float:
        sr, _ = backtest(df, regimes, signals, w)
        return -compute_metrics(sr)["sharpe"]

    # Structured starts: equal weight + each signal in isolation
    structured: list[np.ndarray] = [np.ones(n_sig) / n_sig]
    for i in range(n_sig):
        w = np.zeros(n_sig)
        w[i] = 1.0
        structured.append(w)

    # Random starts
    rng = np.random.default_rng(42)
    randoms = [rng.exponential(1.0, n_sig) for _ in range(n_random)]

    for x0 in structured + randoms:
        x0 = np.abs(x0)
        x0 /= x0.sum() + 1e-10
        try:
            res = minimize(
                objective,
                x0,
                method="Nelder-Mead",
                options={"maxiter": 150, "xatol": 1e-3, "fatol": 1e-3},
            )
            candidate = _normalise_weights(res.x)
            sharpe    = -res.fun
            if sharpe > best_sharpe:
                best_sharpe  = sharpe
                best_weights = candidate
        except Exception:
            pass

    return best_weights


# ──────────────────────────────────────────────────────────────────────────────
# Trade Blotter
# ──────────────────────────────────────────────────────────────────────────────

def generate_trades(df: pd.DataFrame, position: pd.Series) -> pd.DataFrame:
    """Extract individual trades (entry → exit) from a position series."""
    trades      = []
    in_trade    = False
    entry_date  = entry_price = None

    pos = position.dropna()
    for i in range(len(pos)):
        date  = pos.index[i]
        price = float(df.loc[date, "close"]) if date in df.index else np.nan

        if not in_trade and pos.iloc[i] == 1.0:
            in_trade    = True
            entry_date  = date
            entry_price = price

        elif in_trade and pos.iloc[i] == 0.0:
            in_trade = False
            pnl      = (price - entry_price) / (entry_price + 1e-10)
            days     = (date - entry_date).days
            trades.append({
                "Entry Date":     entry_date.strftime("%Y-%m-%d"),
                "Exit Date":      date.strftime("%Y-%m-%d"),
                "Days Held":      days,
                "Entry Price":    round(entry_price, 2),
                "Exit Price":     round(price, 2),
                "Return (%)":     round(pnl * 100, 2),
                "Result":         "Win" if pnl > 0 else "Loss",
            })

    # Still open position
    if in_trade and entry_price is not None:
        current = float(df["close"].iloc[-1])
        pnl     = (current - entry_price) / (entry_price + 1e-10)
        days    = (df.index[-1] - entry_date).days
        trades.append({
            "Entry Date":  entry_date.strftime("%Y-%m-%d"),
            "Exit Date":   "OPEN",
            "Days Held":   days,
            "Entry Price": round(entry_price, 2),
            "Exit Price":  round(current, 2),
            "Return (%)":  round(pnl * 100, 2),
            "Result":      "Open",
        })

    cols = ["Entry Date", "Exit Date", "Days Held",
            "Entry Price", "Exit Price", "Return (%)", "Result"]
    return pd.DataFrame(trades, columns=cols) if trades else pd.DataFrame(columns=cols)


# ──────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_analysis(ticker: str, _progress=None) -> dict | None:
    """
    Full pipeline for one ETF ticker.

    _progress : optional callable(pct: int, msg: str) for live UI updates.
                The leading underscore tells Streamlit's @st.cache_data to
                exclude this argument from the cache key, so the same cached
                result is served regardless of which callback is passed.
    """
    def _upd(pct: int, msg: str) -> None:
        if _progress is not None:
            _progress(pct, msg)

    try:
        _upd(5,  f"Downloading {ticker} price history (3 years)…")
        df = fetch_data(ticker)
        if len(df) < 60:
            return None

        _upd(18, "Computing HMM features (returns · range · vol)…")
        features = compute_features(df)

        _upd(30, "Fitting Gaussian HMM · 5 hidden states…")
        model, regimes, regime_map, scaler = fit_hmm(features)

        _upd(45, "Computing signals: RSI · Momentum · Vol · ADX · EMA…")
        signals = compute_signals(df)

        # ── 2-year lookback window ─────────────────────────────────────────
        lookback_start = df.index[-1] - pd.DateOffset(years=2)
        df_lb          = df[df.index >= lookback_start]
        regimes_lb     = regimes[regimes.index >= lookback_start]
        signals_lb     = signals[signals.index >= lookback_start]

        if len(df_lb) < 60:
            return None

        _upd(58, "Optimising signal weights · maximising Sharpe ratio…")
        weights = optimize_weights(df_lb, regimes_lb, signals_lb)

        _upd(82, "Backtesting strategy on 2-year window…")
        strat_returns, position = backtest(df_lb, regimes_lb, signals_lb, weights)
        metrics = compute_metrics(strat_returns)
        trades  = generate_trades(df_lb, position)

        _upd(93, "Computing regime transition probabilities…")
        current_regime   = str(regimes.iloc[-1])
        current_sig_vals = signals.iloc[-1].values
        w_norm           = _normalise_weights(weights)
        composite        = float(current_sig_vals @ w_norm)

        last_feat    = features.iloc[[-1]].values
        X_last       = scaler.transform(last_feat)
        proba_last   = model.predict_proba(X_last)[0]
        bearish_ids  = [s for s, lbl in regime_map.items() if lbl == "bearish"]
        bearish_prob = float(sum(proba_last[s] for s in bearish_ids))

        # ── Recommendation ────────────────────────────────────────────────
        if current_regime == "bullish":
            recommendation = "BUY" if composite > 0.10 else "NEUTRAL"
        elif current_regime == "bearish":
            recommendation = "SELL"
        else:  # neutral — check transition probability
            recommendation = "SELL" if bearish_prob >= 0.40 else "NEUTRAL"

        _upd(100, "Analysis complete.")

        return {
            "ticker":          ticker,
            "df":              df,
            "df_lb":           df_lb,
            "regimes":         regimes,
            "regimes_lb":      regimes_lb,
            "signals":         signals,
            "signals_lb":      signals_lb,
            "weights":         weights,
            "w_norm":          w_norm,
            "strat_returns":   strat_returns,
            "position":        position,
            "metrics":         metrics,
            "trades":          trades,
            "current_regime":  current_regime,
            "recommendation":  recommendation,
            "composite":       composite,
            "bearish_prob":    bearish_prob,
            "signal_names":    list(signals.columns),
            "current_signals": dict(zip(signals.columns, current_sig_vals)),
        }

    except Exception as exc:
        print(f"[ERROR] {ticker}: {exc}")
        return None
