"""
ETF Regime Trading Dashboard
─────────────────────────────
Streamlit app — dark terminal aesthetic, Plotly charts.
Run: streamlit run app.py
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from etf_engine import (
    ETF_UNIVERSE,
    OBJ_LABELS,
    REGIME_COLORS,
    REGIME_PALETTE,
    run_analysis,
    validate_ticker,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ETF Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS  —  dark terminal theme
# ─────────────────────────────────────────────────────────────────────────────


st.markdown("""
    <style>
    .block-container {
        padding-top: 3rem;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    /* ── Base ─────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
    }
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

    /* ── Cards ────────────────────────────── */
    .card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 18px 20px;
        height: 100%;
    }
    .card-label {
        font-size: 10px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 6px;
    }
    .card-value {
        font-size: 26px;
        font-weight: 700;
        letter-spacing: 1px;
    }
    .pos  { color: #3fb950; }
    .neg  { color: #f85149; }
    .neut { color: #d29922; }
    .blue { color: #58a6ff; }

    /* ── Recommendation ───────────────────── */
    .rec-wrap {
        background: #161b22;
        border-radius: 12px;
        padding: 22px;
        text-align: center;
        height: 100%;
        border: 2px solid;
    }
    .rec-buy   { border-color: #3fb950; }
    .rec-sell  { border-color: #f85149; }
    .rec-neut  { border-color: #d29922; }
    .rec-title { font-size: 10px; color: #8b949e; letter-spacing: 2px; }
    .rec-badge {
        font-size: 36px;
        font-weight: 900;
        letter-spacing: 5px;
        margin: 10px 0 6px;
    }
    .rec-score { font-size: 11px; color: #8b949e; margin-top: 4px; }

    /* ── Regime badge ─────────────────────── */
    .rbadge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .rbadge-bullish { background:rgba(63,185,80,.18);  color:#3fb950; border:1px solid #3fb950; }
    .rbadge-neutral { background:rgba(210,153,34,.18); color:#d29922; border:1px solid #d29922; }
    .rbadge-bearish { background:rgba(248,81,73,.18);  color:#f85149; border:1px solid #f85149; }

    /* ── Metric grid ──────────────────────── */
    .metric-grid { display:flex; gap:12px; margin-bottom:12px; }
    hr.dim { border:none; border-top:1px solid #21262d; margin:14px 0; }

    /* ── Trade blotter ────────────────────── */
    .blotter-header {
        font-size: 10px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 10px;
    }

    /* ── Plotly overrides ─────────────────── */
    .js-plotly-plot .plotly .modebar { background: transparent !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<p style="font-size:20px;font-weight:700;color:#e6edf3;letter-spacing:2px;">'
        '📈 Dashboard Controls</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="dim">', unsafe_allow_html=True)

    # ── Preset universe ───────────────────────────────────────────────────────
    st.markdown(
        '<div class="card-label" style="margin-bottom:4px;">PRESET ETF</div>',
        unsafe_allow_html=True,
    )
    preset_ticker = st.selectbox(
        "Preset ETF",
        ETF_UNIVERSE,
        index=0,
        label_visibility="collapsed",
        key="ticker_select",
    )

    # ── Custom ticker ─────────────────────────────────────────────────────────
    st.markdown(
        '<div class="card-label" style="margin-top:10px;margin-bottom:4px;">'
        'CUSTOM TICKER (overrides preset)</div>',
        unsafe_allow_html=True,
    )
    custom_input = st.text_input(
        "Custom ticker",
        value="",
        placeholder="e.g. SPY, IWM, ARKK, GLD…",
        label_visibility="collapsed",
        key="custom_ticker_input",
        max_chars=10,
    )

    # Resolve active ticker
    custom_clean = custom_input.strip().upper()
    ticker       = custom_clean if custom_clean else preset_ticker

    if custom_clean:
        st.markdown(
            f'<div style="font-size:11px;color:#58a6ff;margin-top:2px;">'
            f'Using custom: <b>{ticker}</b></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="dim">', unsafe_allow_html=True)
    st.markdown(
        """
        **Strategy logic**
        - Regime: GaussianHMM · 5 states
        - Signals: RSI · Momentum · Vol · ADX · EMA · Stoch · Bollinger · CMF
        - Entry: bullish regime ∧ score > 0.03
        - Exit: bearish regime (or neutral + bearish prob ≥ 30%)
        - Lookback: 2 years
        """,
        unsafe_allow_html=True,
    )

    # ── Optimisation objective ────────────────────────────────────────────────
    st.markdown('<hr class="dim">', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-label" style="margin-bottom:4px;">OPTIMISATION OBJECTIVE</div>',
        unsafe_allow_html=True,
    )
    _OBJ_OPTIONS  = list(OBJ_LABELS.keys())
    _OBJ_DISPLAY  = {k: f"Max {v.title()}" for k, v in OBJ_LABELS.items()}
    objective = st.selectbox(
        "Optimisation objective",
        options=_OBJ_OPTIONS,
        format_func=lambda x: _OBJ_DISPLAY[x],
        index=0,
        label_visibility="collapsed",
        key="objective_select",
    )
    st.markdown(
        f'<div style="font-size:10px;color:#8b949e;margin-top:2px;">'
        f'Weights are tuned to maximise {OBJ_LABELS[objective]}</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="dim">', unsafe_allow_html=True)
    refresh = st.button("⟳  Refresh Analysis", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Ticker validation  (only for custom tickers — fast 5-day check)
# @st.cache_data is safe here: validate_ticker makes no Streamlit element calls.
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3_600, show_spinner=False)
def _cached_validate(sym: str) -> tuple[bool, str]:
    return validate_ticker(sym)


if custom_clean and custom_clean not in ETF_UNIVERSE:
    with st.spinner(f"Validating {ticker}…"):
        is_valid, err_msg = _cached_validate(ticker)
    if not is_valid:
        st.error(
            f"**Invalid ticker: {ticker}**\n\n"
            f"{err_msg}\n\n"
            f"*Preset universe: {' · '.join(ETF_UNIVERSE)}*"
        )
        st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Process-global version counter  (shared refresh state across all sessions)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def _get_version_state() -> dict:
    """Initialised once per process; persists for the lifetime of the server."""
    return {"versions": {}, "lock": threading.Lock()}

_vstate = _get_version_state()

if refresh:
    with _vstate["lock"]:
        _vstate["versions"][ticker] = _vstate["versions"].get(ticker, 0) + 1

with _vstate["lock"]:
    _cache_v = _vstate["versions"].get(ticker, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Manual analysis cache  (replaces @st.cache_data for the compute path)
#
# WHY NOT @st.cache_data here:
#   @st.cache_data records every Streamlit element call made during the cached
#   function — including those made via progress callbacks — as "messages" to
#   replay on future cache hits. On a cache hit from a DIFFERENT session,
#   Streamlit tries to replay those calls using element IDs from the original
#   session, which no longer exist → KeyError → CacheReplayClosureError.
#
#   A plain dict in @st.cache_resource has NO replay mechanism, so progress
#   callbacks referencing session-local UI elements (st.progress, st.empty)
#   work safely across concurrent users.
# ─────────────────────────────────────────────────────────────────────────────
_CACHE_TTL  = 3_600   # seconds — matches old ttl=3600
_CACHE_MAX  = 50      # max entries before LRU eviction

@st.cache_resource
def _get_analysis_store() -> dict:
    """Process-global result store. Initialised once per server process."""
    return {"results": {}, "lock": threading.Lock()}

_astore = _get_analysis_store()


def _cache_get(sym: str, cv: int, obj: str) -> dict | None:
    """Return cached result if present and not expired, else None."""
    key = (sym, cv, obj)
    with _astore["lock"]:
        entry = _astore["results"].get(key)
        if entry is None:
            return None
        result, ts = entry
        if time.time() - ts > _CACHE_TTL:
            del _astore["results"][key]
            return None
        return result


def _cache_put(sym: str, cv: int, obj: str, result: dict | None) -> None:
    """Store result; evict expired and oldest-over-limit entries."""
    if result is None:
        return
    key = (sym, cv, obj)
    with _astore["lock"]:
        now = time.time()
        stale = [k for k, (_, ts) in _astore["results"].items() if now - ts > _CACHE_TTL]
        for k in stale:
            del _astore["results"][k]
        while len(_astore["results"]) >= _CACHE_MAX:
            oldest = min(_astore["results"], key=lambda k: _astore["results"][k][1])
            del _astore["results"][oldest]
        _astore["results"][key] = (result, now)


# ─────────────────────────────────────────────────────────────────────────────
# Compute or serve from cache
# ─────────────────────────────────────────────────────────────────────────────
result = _cache_get(ticker, _cache_v, objective)

if result is None:
    # Cache miss — show staged progress bar and run analysis directly.
    # The progress callback captures session-local UI elements (_pbar,
    # _status_txt). Because we call run_analysis() directly (NOT inside a
    # @st.cache_data function), there is no replay mechanism and no cross-
    # session element-ID collision.
    _progress_slot = st.empty()
    with _progress_slot.container():
        st.markdown("""
            <style>
            .block-container {
                padding-top: 3rem;
            }
            </style>
            """, unsafe_allow_html=True)
        
        st.markdown(
            f'<div class="card-label" style="margin-bottom:6px;">'
            f'RUNNING ANALYSIS — {ticker}</div>',
            unsafe_allow_html=True,
        )
        _pbar       = st.progress(0, text="Initialising…")
        _status_txt = st.empty()

    def _progress_callback(pct: int, msg: str) -> None:
        _pbar.progress(
            min(pct, 100) / 100,
            text=f"{'✅' if pct >= 100 else '⏳'}  {msg}",
        )
        stage = (
            f"Step {max(1, round(pct / 100 * 6))}/6"
            if pct < 100 else "Complete"
        )
        _status_txt.markdown(
            f'<div style="font-size:10px;color:#8b949e;font-family:monospace;">'
            f'{stage} · {pct}%</div>',
            unsafe_allow_html=True,
        )

    result = run_analysis(ticker, _progress=_progress_callback, objective=objective)
    _progress_slot.empty()
    _cache_put(ticker, _cache_v, objective, result)


# ─────────────────────────────────────────────────────────────────────────────
# Error handling
# ─────────────────────────────────────────────────────────────────────────────
if result is None:
    st.error(
        f"**{ticker}** — analysis failed.  \n\n"
        "Possible causes: insufficient price history (< 2 years), "
        "the symbol is not a US-listed instrument, or a temporary data error.  \n\n"
        f"*Preset universe: {' · '.join(ETF_UNIVERSE)}*"
    )
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Unpack
# ─────────────────────────────────────────────────────────────────────────────
df            = result["df"]
df_lb         = result["df_lb"]
regimes       = result["regimes"]
regimes_lb    = result["regimes_lb"]
signals       = result["signals"]
signals_lb    = result["signals_lb"]
weights       = result["weights"]
w_norm        = result["w_norm"]
strat_returns = result["strat_returns"]
position      = result["position"]
metrics       = result["metrics"]
trades        = result["trades"]
recommendation   = result["recommendation"]
current_regime   = result["current_regime"]
composite        = result["composite"]
bearish_prob     = result["bearish_prob"]
signal_names     = result["signal_names"]
current_signals  = result["current_signals"]
result_objective = result.get("objective", "sharpe")

current_price = float(df["close"].iloc[-1])
prev_price    = float(df["close"].iloc[-2])
price_chg     = (current_price - prev_price) / (prev_price + 1e-10)
price_color   = "#3fb950" if price_chg >= 0 else "#f85149"


# ─────────────────────────────────────────────────────────────────────────────
# ── HEADER ────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="display:flex;align-items:baseline;gap:18px;margin-bottom:4px;">
        <span style="font-size:22px;font-weight:800;color:#e6edf3;letter-spacing:2px;">
            ETF TRADING DASHBOARD
        </span>
        <span style="font-size:18px;color:#58a6ff;font-weight:700;">{ticker}</span>
        <span style="font-size:20px;color:#e6edf3;font-weight:600;">
            ${current_price:,.2f}
        </span>
        <span style="font-size:16px;color:{price_color};font-weight:600;">
            {price_chg:+.2%}
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown('<hr class="dim">', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── ROW 1: Recommendation | Regime | Signals ──────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
col_rec, col_reg, col_sig = st.columns([1.1, 1.1, 2.2], gap="medium")

# ── Recommendation ────────────────────────────────────────────────────────────
rec_css = {"BUY": "rec-buy", "SELL": "rec-sell", "NEUTRAL": "rec-neut"}[recommendation]
rec_col = {"BUY": "#3fb950", "SELL": "#f85149", "NEUTRAL": "#d29922"}[recommendation]
rec_ico = {"BUY": "▲", "SELL": "▼", "NEUTRAL": "◆"}[recommendation]

if recommendation == "BUY":
    rec_rationale = "Bullish regime · signals aligned"
elif recommendation == "SELL" and current_regime == "bearish":
    rec_rationale = "Bearish regime confirmed · exit"
elif recommendation == "SELL":
    rec_rationale = f"Regime transitioning · {bearish_prob:.0%} bearish prob"
else:
    rec_rationale = "Await clearer regime signal"

with col_rec:
    st.markdown(
        f"""
        <div class="rec-wrap {rec_css}">
            <div class="rec-title">RECOMMENDATION</div>
            <div class="rec-badge" style="color:{rec_col};">{rec_ico} {recommendation}</div>
            <div class="rec-score">
                Composite score&nbsp;
                <span style="color:#e6edf3;font-size:13px;">{composite:+.3f}</span>
            </div>
            <div style="font-size:10px;color:{rec_col};margin-top:6px;letter-spacing:0.5px;">
                {rec_rationale}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Regime ────────────────────────────────────────────────────────────────────
bull_pct = (regimes_lb == "bullish").mean() * 100
bear_pct = (regimes_lb == "bearish").mean() * 100
neut_pct = (regimes_lb == "neutral").mean() * 100

bp_pct   = bearish_prob * 100
bp_color = "#f85149" if bearish_prob >= 0.30 else ("#d29922" if bearish_prob >= 0.15 else "#8b949e")
bp_label = "HIGH — exit risk elevated" if bearish_prob >= 0.30 else (
           "MODERATE" if bearish_prob >= 0.15 else "LOW")

# ── IMPORTANT: Streamlit's st.markdown runs content through a Markdown parser
# before rendering HTML. Any line with ≥ 4 spaces of leading whitespace is
# treated as a Markdown code block, so closing </div> tags on indented lines
# render as literal text. Fix: build HTML via single-line string concatenation
# so no line has leading whitespace. Never use multiline f-strings with deep
# indentation for HTML passed to st.markdown(unsafe_allow_html=True).
bp_html = (
    f'<hr class="dim">'
    f'<div class="card-label">BEARISH TRANSITION PROBABILITY</div>'
    f'<div style="margin-top:6px;">'
    f'<div style="background:#21262d;border-radius:4px;height:7px;overflow:hidden;">'
    f'<div style="width:{min(bp_pct,100):.0f}%;height:100%;background:{bp_color};'
    f'border-radius:4px;transition:width .4s;"></div>'
    f'</div>'
    f'<div style="display:flex;justify-content:space-between;margin-top:4px;">'
    f'<span style="font-size:11px;color:{bp_color};">{bp_label}</span>'
    f'<span style="font-size:11px;color:#e6edf3;font-family:monospace;">{bp_pct:.0f}%</span>'
    f'</div>'
    f'</div>'
)

with col_reg:
    st.markdown(
        f'<div class="card">'
        f'<div class="card-label">CURRENT MARKET REGIME</div>'
        f'<div style="margin:10px 0 12px;">'
        f'<span class="rbadge rbadge-{current_regime}">{current_regime.upper()}</span>'
        f'</div>'
        f'<hr class="dim">'
        f'<div class="card-label">2-Year Regime Distribution</div>'
        f'<div style="display:flex;gap:14px;margin-top:8px;flex-wrap:wrap;">'
        f'<span style="color:#3fb950;font-size:12px;">▲ Bull {bull_pct:.0f}%</span>'
        f'<span style="color:#d29922;font-size:12px;">◆ Neut {neut_pct:.0f}%</span>'
        f'<span style="color:#f85149;font-size:12px;">▼ Bear {bear_pct:.0f}%</span>'
        f'</div>'
        f'{bp_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── Signal bar chart ──────────────────────────────────────────────────────────
with col_sig:
    sig_vals    = list(current_signals.values())
    sig_labels  = [n.upper() for n in signal_names]
    bar_colors  = ["#3fb950" if v >= 0 else "#f85149" for v in sig_vals]
    weight_pcts = [f"{w:.0%}" for w in w_norm]

    fig_sig = go.Figure(
        go.Bar(
            x=sig_labels,
            y=sig_vals,
            marker_color=bar_colors,
            text=[f"{v:+.2f}" for v in sig_vals],
            textposition="outside",
            textfont=dict(size=10, color="#c9d1d9"),
            customdata=weight_pcts,
            hovertemplate="<b>%{x}</b><br>Value: %{y:.3f}<br>Weight: %{customdata}<extra></extra>",
        )
    )
    fig_sig.add_hline(y=0, line_color="#30363d", line_width=1)
    fig_sig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=148,
        margin=dict(l=4, r=4, t=22, b=4),
        title=dict(
            text="SIGNAL SCORES  (weights: " + "  ".join(
                f"{n.upper()} {w:.0%}" for n, w in zip(signal_names, w_norm)
            ) + ")",
            font=dict(size=9, color="#8b949e"),
            x=0,
        ),
        xaxis=dict(showgrid=False, tickfont=dict(size=11)),
        yaxis=dict(showgrid=False, range=[-1.3, 1.3], zeroline=False),
        showlegend=False,
    )
    st.plotly_chart(fig_sig, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# ── ROW 2: Performance metrics ────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="dim">', unsafe_allow_html=True)
_obj_display_name = OBJ_LABELS.get(result_objective, "sharpe ratio").title()
st.markdown(
    f'<div class="card-label" style="font-size:11px;margin-bottom:10px;">'
    f'PERFORMANCE METRICS — 2-YEAR BACKTEST'
    f'<span style="color:#58a6ff;margin-left:10px;">· OPTIMISED FOR {_obj_display_name.upper()}</span>'
    f'</div>',
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4, gap="medium")

tr      = metrics["total_return"]
ann_ret = metrics["annual_return"]
sr      = metrics["sharpe"]
sortino = metrics["sortino"]
calmar  = metrics["calmar"]
wr      = metrics["win_rate"]
mdd     = metrics["max_drawdown"]

def _metric_card(label: str, value_str: str, css_class: str, badge: str = "") -> str:
    badge_html = (
        f'<div style="font-size:9px;color:#58a6ff;margin-top:4px;letter-spacing:1px;">'
        f'{badge}</div>'
    ) if badge else ""
    return (
        f'<div class="card" style="text-align:center;">'
        f'<div class="card-label">{label}</div>'
        f'<div class="card-value {css_class}">{value_str}</div>'
        f'{badge_html}'
        f'</div>'
    )

# Card 1 — Total return (always shown)
with m1:
    st.markdown(
        _metric_card("Total Return", f"{tr:+.1%}", "pos" if tr >= 0 else "neg"),
        unsafe_allow_html=True,
    )

# Card 2 — Optimised metric (dynamic)
with m2:
    if result_objective == "sortino":
        s_cls = "pos" if sortino > 1 else ("neg" if sortino < 0 else "neut")
        st.markdown(_metric_card("Sortino Ratio", f"{sortino:.2f}", s_cls, "OPTIMISED"), unsafe_allow_html=True)
    elif result_objective == "calmar":
        c_cls = "pos" if calmar > 1 else ("neg" if calmar < 0 else "neut")
        st.markdown(_metric_card("Calmar Ratio", f"{calmar:.2f}", c_cls, "OPTIMISED"), unsafe_allow_html=True)
    elif result_objective == "annual_return":
        a_cls = "pos" if ann_ret >= 0 else "neg"
        st.markdown(_metric_card("Annual Return", f"{ann_ret:+.1%}", a_cls, "OPTIMISED"), unsafe_allow_html=True)
    else:  # sharpe (default)
        sr_cls = "pos" if sr > 1 else ("neg" if sr < 0 else "neut")
        st.markdown(_metric_card("Sharpe Ratio", f"{sr:.2f}", sr_cls, "OPTIMISED"), unsafe_allow_html=True)

# Card 3 — Win rate (always shown)
with m3:
    wr_cls = "pos" if wr >= 0.5 else ("neg" if wr < 0.4 else "neut")
    st.markdown(_metric_card("Win Rate", f"{wr:.1%}", wr_cls), unsafe_allow_html=True)

# Card 4 — Max drawdown (always shown)
with m4:
    st.markdown(_metric_card("Max Drawdown", f"{mdd:.1%}", "neg"), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── ROW 3: Candlestick chart with regime shading ──────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="dim">', unsafe_allow_html=True)
st.markdown(
    '<div class="card-label" style="font-size:11px;margin-bottom:8px;">'
    'PRICE ACTION — REGIME BACKGROUND · ENTRY / EXIT MARKERS</div>',
    unsafe_allow_html=True,
)


def _regime_shading(fig: go.Figure, regimes_s: pd.Series) -> None:
    """Paint vertical bands by regime."""
    if regimes_s.empty:
        return
    rdf = pd.DataFrame({"regime": regimes_s})
    rdf["grp"] = (rdf["regime"] != rdf["regime"].shift()).cumsum()
    for _, grp in rdf.groupby("grp", sort=False):
        regime = grp["regime"].iloc[0]
        fig.add_vrect(
            x0=grp.index[0],
            x1=grp.index[-1],
            fillcolor=REGIME_COLORS.get(regime, "rgba(0,0,0,0)"),
            opacity=1,
            layer="below",
            line_width=0,
        )


def build_price_chart(
    df_1y: pd.DataFrame,
    regimes_1y: pd.Series,
    position: pd.Series,
) -> go.Figure:
    fig = go.Figure()

    # ── Bollinger Bands (20d, 2σ) — draw first so candles sit on top ─────────
    bb_mid   = df_1y["close"].rolling(20, min_periods=10).mean()
    bb_sigma = df_1y["close"].rolling(20, min_periods=10).std()
    bb_upper = bb_mid + 2 * bb_sigma
    bb_lower = bb_mid - 2 * bb_sigma

    fig.add_trace(go.Scatter(
        x=df_1y.index, y=bb_lower,
        line=dict(color="rgba(188,140,255,0.28)", width=1),
        name="BB Lower", showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=df_1y.index, y=bb_upper,
        line=dict(color="rgba(188,140,255,0.28)", width=1),
        fill="tonexty", fillcolor="rgba(188,140,255,0.055)",
        name="BB (20d, 2σ)", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=df_1y.index, y=bb_mid,
        line=dict(color="rgba(188,140,255,0.45)", width=1, dash="dot"),
        name="SMA 20", showlegend=False, hoverinfo="skip",
    ))

    # ── EMA overlays (20 + 50 — standard trader pair) ─────────────────────────
    for span, col in [(20, "#58a6ff"), (50, "#d29922")]:
        fig.add_trace(go.Scatter(
            x=df_1y.index,
            y=df_1y["close"].ewm(span=span, adjust=False).mean(),
            name=f"EMA {span}",
            line=dict(color=col, width=1.3),
            opacity=0.9, hoverinfo="skip",
        ))

    # ── Candlestick — rendered above all overlays ─────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df_1y.index,
        open=df_1y["open"], high=df_1y["high"],
        low=df_1y["low"],   close=df_1y["close"],
        name="OHLC",
        increasing=dict(line=dict(color="#3fb950", width=1), fillcolor="#3fb950"),
        decreasing=dict(line=dict(color="#f85149", width=1), fillcolor="#f85149"),
        whiskerwidth=0.3,
    ))

    # ── Current price dotted line ─────────────────────────────────────────────
    last_close = float(df_1y["close"].iloc[-1])
    fig.add_hline(
        y=last_close,
        line=dict(color="#e6edf3", width=1, dash="dot"),
        opacity=0.55,
        annotation_text=f"  ${last_close:,.2f}",
        annotation_position="right",
        annotation_font=dict(size=11, color="#e6edf3"),
    )

    # ── Entry / exit markers ──────────────────────────────────────────────────
    pos_r   = position.reindex(df_1y.index).fillna(0.0)
    prev    = pos_r.shift(1, fill_value=0.0)
    entries = pos_r.index[(pos_r == 1.0) & (prev == 0.0)]
    exits   = pos_r.index[(pos_r == 0.0) & (prev == 1.0)]

    if len(entries):
        fig.add_trace(go.Scatter(
            x=entries,
            y=df_1y.loc[entries, "low"] * 0.986,
            mode="markers",
            name="Entry ▲",
            marker=dict(symbol="triangle-up", size=13, color="#3fb950",
                        line=dict(color="#e6edf3", width=0.8)),
            hovertemplate="Entry: %{x|%Y-%m-%d}<extra></extra>",
        ))

    if len(exits):
        fig.add_trace(go.Scatter(
            x=exits,
            y=df_1y.loc[exits, "high"] * 1.014,
            mode="markers",
            name="Exit ▼",
            marker=dict(symbol="triangle-down", size=13, color="#f85149",
                        line=dict(color="#e6edf3", width=0.8)),
            hovertemplate="Exit: %{x|%Y-%m-%d}<extra></extra>",
        ))

    # ── Regime shading ────────────────────────────────────────────────────────
    _regime_shading(fig, regimes_1y)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        height=580,
        margin=dict(l=10, r=70, t=10, b=10),
        legend=dict(
            orientation="h", y=1.01, x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10),
        ),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridcolor="#1c2128", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#1c2128", side="right", tickprefix="$"),
    )

    return fig


chart = build_price_chart(df_lb, regimes_lb, position)
st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": True, "displaylogo": False})

# ── Regime legend key ─────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="display:flex;flex-wrap:wrap;gap:18px;margin-top:-8px;margin-bottom:4px;font-size:11px;color:#8b949e;">
        <span><span style="display:inline-block;width:12px;height:12px;background:rgba(0,200,100,0.30);
              border-radius:2px;margin-right:4px;"></span>Bullish regime</span>
        <span><span style="display:inline-block;width:12px;height:12px;background:rgba(255,80,80,0.30);
              border-radius:2px;margin-right:4px;"></span>Bearish regime</span>
        <span><span style="display:inline-block;width:12px;height:12px;background:rgba(210,153,34,0.20);
              border-radius:2px;margin-right:4px;"></span>Neutral regime</span>
        <span style="color:#bc8cff;">▬ Bollinger Bands (20d, 2σ)</span>
        <span style="color:#58a6ff;">▬ EMA 20</span>
        <span style="color:#d29922;">▬ EMA 50</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# ── ROW 4: Cumulative returns vs Buy & Hold ────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="dim">', unsafe_allow_html=True)
st.markdown(
    '<div class="card-label" style="font-size:11px;margin-bottom:8px;">'
    'CUMULATIVE RETURNS — STRATEGY vs BUY &amp; HOLD</div>',
    unsafe_allow_html=True,
)

if len(strat_returns) > 1:
    cum_strat = (1 + strat_returns).cumprod()
    bh_ret    = df_lb["close"].pct_change().dropna().reindex(strat_returns.index)
    cum_bh    = (1 + bh_ret.fillna(0)).cumprod()

    final_strat = cum_strat.iloc[-1]
    final_bh    = cum_bh.iloc[-1]

    fig_cum = go.Figure()
    fig_cum.add_trace(
        go.Scatter(
            x=cum_strat.index, y=cum_strat,
            name=f"Strategy ({final_strat - 1:+.1%})",
            line=dict(color="#58a6ff", width=2.2),
            fill="tozeroy",
            fillcolor="rgba(88,166,255,0.07)",
            hovertemplate="%{x|%Y-%m-%d}  %{y:.3f}<extra>Strategy</extra>",
        )
    )
    fig_cum.add_trace(
        go.Scatter(
            x=cum_bh.index, y=cum_bh,
            name=f"Buy & Hold ({final_bh - 1:+.1%})",
            line=dict(color="#8b949e", width=1.5, dash="dash"),
            hovertemplate="%{x|%Y-%m-%d}  %{y:.3f}<extra>Buy & Hold</extra>",
        )
    )
    fig_cum.add_hline(y=1.0, line_color="#30363d", line_dash="dot", line_width=1)

    fig_cum.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        height=230,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        xaxis=dict(showgrid=True, gridcolor="#1c2128"),
        yaxis=dict(showgrid=True, gridcolor="#1c2128", side="right", tickformat=".2f"),
        hovermode="x unified",
    )
    st.plotly_chart(fig_cum, use_container_width=True, config={"displayModeBar": False})
else:
    st.info("Insufficient trades to plot cumulative returns.")


# ─────────────────────────────────────────────────────────────────────────────
# ── ROW 5: Trade Blotter ──────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="dim">', unsafe_allow_html=True)
st.markdown(
    '<div class="card-label" style="font-size:11px;margin-bottom:12px;">'
    'TRADE BLOTTER — 2-YEAR LOOKBACK</div>',
    unsafe_allow_html=True,
)

if trades.empty:
    st.info(
        "No trades were executed in the 2-year lookback window.  "
        "Bullish regime + positive composite signal was never triggered simultaneously."
    )
else:
    # ── Summary mini-metrics ──────────────────────────────────────────────────
    n_total = len(trades)
    n_win   = int((trades["Result"] == "Win").sum())
    n_loss  = int((trades["Result"] == "Loss").sum())
    n_open  = int((trades["Result"] == "Open").sum())
    avg_ret = trades["Return (%)"].mean()

    bc1, bc2, bc3, bc4, bc5 = st.columns(5, gap="medium")
    with bc1:
        st.markdown(_metric_card("Total Trades", str(n_total), "blue"), unsafe_allow_html=True)
    with bc2:
        st.markdown(_metric_card("Wins", str(n_win), "pos"), unsafe_allow_html=True)
    with bc3:
        st.markdown(_metric_card("Losses", str(n_loss), "neg"), unsafe_allow_html=True)
    with bc4:
        st.markdown(_metric_card("Open", str(n_open), "blue"), unsafe_allow_html=True)
    with bc5:
        st.markdown(
            _metric_card("Avg Return", f"{avg_ret:+.2f}%", "pos" if avg_ret >= 0 else "neg"),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Styled table ──────────────────────────────────────────────────────────
    display_df = (
        trades
        .sort_values("Entry Date", ascending=False)
        .reset_index(drop=True)
    )

    def _color_result(val: str) -> str:
        return {
            "Win":  "color: #3fb950; font-weight:600;",
            "Loss": "color: #f85149; font-weight:600;",
            "Open": "color: #58a6ff; font-weight:600;",
        }.get(val, "")

    def _color_return(val: float) -> str:
        if isinstance(val, (int, float)):
            return "color: #3fb950;" if val > 0 else "color: #f85149;"
        return ""

    styled = (
        display_df.style
        .map(_color_result, subset=["Result"])
        .map(_color_return, subset=["Return (%)"])
        .set_properties(**{
            "background-color": "#161b22",
            "border":           "1px solid #21262d",
            "font-size":        "12px",
        })
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#21262d"),
                    ("color", "#8b949e"),
                    ("font-size", "10px"),
                    ("text-transform", "uppercase"),
                    ("letter-spacing", "1px"),
                ],
            },
            {
                "selector": "tr:hover td",
                "props": [("background-color", "#1c2128")],
            },
        ])
        .format({"Return (%)": "{:+.2f}%", "Entry Price": "${:.2f}", "Exit Price": "${:.2f}"})
    )

    st.dataframe(styled, use_container_width=True, height=min(420, 60 + 36 * len(display_df)))


# ─────────────────────────────────────────────────────────────────────────────
# ── Footer ────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="dim">', unsafe_allow_html=True)
st.markdown(
    '<div style="font-size:10px;color:#484f58;text-align:center;">'
    '<br>This is for educational purposes and should not be treated as investment advice.</br>'
    '<br>Data provided via yfinance · Training conducted via hmmlearn GaussianHMM · Signals optimised via SciPy</br>'
    '<br>Email the author at ryangoh@outlook.com for feedback</br>'
    '</div>',
    unsafe_allow_html=True,
)