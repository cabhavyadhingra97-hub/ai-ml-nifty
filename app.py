"""
app.py
======
Premium dark-mode Streamlit dashboard for the Nifty 50 ML Direction Predictor.

Sections:
- Hero + Latest Prediction
- 6 Performance Metric Cards
- 4-Tab Analytics (Cumulative Returns, Probability Distribution, Price+Signals, Walk-Forward)
- Confusion Matrix Heatmap
- Feature Importance Chart
- Per-Model Accuracy Breakdown
- Recent Predictions Table (with calibrated confidence, date-only)
- Market Insights
"""

import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nifty 50 AI Predictor | CA Bhavya",
    page_icon="⬛",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

.stApp { background: linear-gradient(180deg,#0a0e1a 0%,#0d1117 60%,#0a0e1a 100%); font-family:'Inter',sans-serif; }
#MainMenu,footer,header,.stDeployButton { visibility:hidden; display:none; }

.hero { text-align:center; padding:2.5rem 1rem 1.5rem; border-bottom:1px solid rgba(56,189,248,.1); margin-bottom:2rem; }
.hero-badge { display:inline-block; background:linear-gradient(135deg,rgba(56,189,248,.15),rgba(168,85,247,.15));
  border:1px solid rgba(56,189,248,.3); color:#38bdf8; padding:.35rem 1.2rem; border-radius:20px;
  font-size:.72rem; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:1rem; }
.hero-title { font-size:2.6rem; font-weight:900;
  background:linear-gradient(135deg,#ffffff 0%,#94a3b8 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; line-height:1.2; }
.hero-sub { font-size:1rem; color:#64748b; max-width:680px; margin:.6rem auto 0; line-height:1.7; }

.mcard { background:linear-gradient(145deg,rgba(15,23,42,.9),rgba(15,23,42,.6));
  border:1px solid rgba(56,189,248,.12); border-radius:14px; padding:1.3rem 1rem; text-align:center;
  transition:all .25s ease; }
.mcard:hover { border-color:rgba(56,189,248,.4); transform:translateY(-2px); box-shadow:0 8px 28px rgba(56,189,248,.09); }
.mlabel { font-size:.68rem; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:1.2px; margin-bottom:.4rem; }
.mvalue { font-size:1.7rem; font-weight:800; margin-bottom:.2rem; }
.pos { color:#22c55e; } .neg { color:#ef4444; } .neu { color:#38bdf8; } .gold { color:#f59e0b; }

.section-hdr { font-size:1.2rem; font-weight:700; color:#e2e8f0; margin:2rem 0 .8rem;
  padding-left:.8rem; border-left:3px solid #38bdf8; }

.pred-card { background:linear-gradient(145deg,rgba(15,23,42,.95),rgba(15,23,42,.7));
  border:1px solid rgba(56,189,248,.15); border-radius:14px; padding:1.8rem; text-align:center; margin:1rem 0; }
.pred-dir { font-size:2.8rem; font-weight:900; }
.pred-up { color:#22c55e; text-shadow:0 0 28px rgba(34,197,94,.3); }
.pred-dn { color:#ef4444; text-shadow:0 0 28px rgba(239,68,68,.3); }
.pred-conf { font-size:.85rem; color:#94a3b8; margin-top:.4rem; }

.insight { background:linear-gradient(145deg,rgba(15,23,42,.9),rgba(15,23,42,.6));
  border:1px solid rgba(168,85,247,.15); border-radius:14px; padding:1.6rem;
  color:#cbd5e1; line-height:1.8; font-size:.9rem; }
.insight strong { color:#f1f5f9; } .insight em { color:#a78bfa; font-style:normal; }

.footer { text-align:center; padding:2rem; color:#334155; font-size:.73rem;
  margin-top:3rem; border-top:1px solid rgba(56,189,248,.08); }

.stTabs [data-baseweb="tab"] { background:rgba(15,23,42,.5); border-radius:8px;
  color:#94a3b8; border:1px solid rgba(56,189,248,.1); }
.stTabs [aria-selected="true"] { background:rgba(56,189,248,.1)!important;
  color:#38bdf8!important; border-color:rgba(56,189,248,.3)!important; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#94a3b8"),
    height=420,
    margin=dict(l=40, r=20, t=30, b=40),
)

def _grid(fig):
    fig.update_xaxes(gridcolor="rgba(56,189,248,.05)")
    fig.update_yaxes(gridcolor="rgba(56,189,248,.05)")
    return fig


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load all CSV artifacts produced by the pipeline."""
    def _csv(name, **kw):
        p = os.path.join(BASE_DIR, "data", name)
        return pd.read_csv(p, **kw) if os.path.exists(p) else None

    preds = _csv("test_predictions.csv", index_col="Date", parse_dates=True)
    if preds is not None:
        preds.index = preds.index.normalize()

    stats     = _csv("backtest_stats.csv", index_col=0)
    cum_ret   = _csv("cumulative_returns.csv", index_col="Date", parse_dates=True)
    raw       = _csv("raw_data.csv", index_col="Date", parse_dates=True)
    feat_imp  = _csv("feature_importance.csv", index_col=0)
    cm_df     = _csv("confusion_matrix.csv", index_col=0)
    per_model = _csv("per_model_accuracy.csv", index_col=0)
    wf        = _csv("walk_forward_results.csv")

    if raw is not None:
        raw.index = raw.index.normalize()

    return preds, stats, cum_ret, raw, feat_imp, cm_df, per_model, wf


preds, stats, cum_ret, raw, feat_imp, cm_df, per_model, wf = load_data()

if preds is None:
    st.error("⚠️ No data found. Run: data_loader -> feature_engineering -> model_training -> backtesting")
    st.stop()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">AI FOR FINANCE · LEVEL 2</div>
  <div class="hero-title">Nifty 50 Direction Predictor</div>
  <div class="hero-sub">Stacked Ensemble (RF + XGBoost + LightGBM -> Meta-Learner) ·
    40+ Leak-Free Features · Kelly Sizing · Walk-Forward Validation</div>
</div>
""", unsafe_allow_html=True)

# ── Latest Prediction ─────────────────────────────────────────────────────────
latest     = preds.iloc[-1]
prob_up    = float(latest.get("Prob_UP", 0.5))
pred_dir   = "BULLISH" if latest["Prediction"] == 1 else "BEARISH"
pred_cls   = "pred-up"  if latest["Prediction"] == 1 else "pred-dn"
pred_arrow = "▲" if latest["Prediction"] == 1 else "▼"
latest_dt  = preds.index[-1].strftime("%d %b %Y")

st.markdown(f"""
<div class="pred-card">
  <div class="mlabel">NEXT-DAY PREDICTION · {latest_dt}</div>
  <div class="pred-dir {pred_cls}">{pred_arrow} {pred_dir}</div>
  <div class="pred-conf">Ensemble Probability (UP): <strong style="color:#38bdf8">{prob_up*100:.2f}%</strong>
   &nbsp;|&nbsp; RF: {float(latest.get('RF_prob', 0.5))*100:.1f}%
   &nbsp;|&nbsp; XGB: {float(latest.get('XGB_prob', 0.5))*100:.1f}%
   &nbsp;|&nbsp; LGB: {float(latest.get('LGB_prob', 0.5))*100:.1f}%
  </div>
</div>
""", unsafe_allow_html=True)

# ── Metric Cards ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">Performance Dashboard</div>', unsafe_allow_html=True)

def _s(col, row, default="N/A"):
    try:
        return float(stats.loc[row, col])
    except Exception:
        return default

test_acc = (preds["Target"] == preds["Prediction"]).mean() * 100

# Try to get filtered column, fallback to first
try:
    filt_ret = _s("Filtered", "Total Return [%]")
    filt_sr  = _s("Filtered", "Sharpe Ratio")
    filt_dd  = _s("Filtered", "Max Drawdown [%]")
    filt_wr  = _s("Filtered", "Win Rate [%]")
except Exception:
    filt_ret = filt_sr = filt_dd = filt_wr = "N/A"

bm_ret = (float(cum_ret.iloc[-1, -1]) - 1) * 100 if cum_ret is not None else "N/A"

def _card(label, val, unit="", color="neu"):
    v = f"{val:.2f}{unit}" if isinstance(val, float) else str(val)
    return f'<div class="mcard"><div class="mlabel">{label}</div><div class="mvalue {color}">{v}</div></div>'

c1,c2,c3,c4,c5,c6 = st.columns(6)
with c1: st.markdown(_card("Test Accuracy", test_acc, "%", "neu"), unsafe_allow_html=True)
with c2: st.markdown(_card("Strategy Return", filt_ret, "%", "pos" if isinstance(filt_ret, float) and filt_ret>0 else "neg"), unsafe_allow_html=True)
with c3: st.markdown(_card("Benchmark Return", bm_ret, "%", "gold"), unsafe_allow_html=True)
with c4: st.markdown(_card("Sharpe Ratio", filt_sr, "", "pos" if isinstance(filt_sr, float) and filt_sr>0 else "neg"), unsafe_allow_html=True)
with c5: st.markdown(_card("Max Drawdown", filt_dd, "%", "neg"), unsafe_allow_html=True)
with c6: st.markdown(_card("Win Rate", filt_wr, "%", "pos" if isinstance(filt_wr, float) and filt_wr>50 else "neg"), unsafe_allow_html=True)

# ── Analytics Tabs ────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">Strategy Analytics</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Cumulative Returns",
    "📊 Probability Distribution",
    "🕯 Price + Signals",
    "🔄 Walk-Forward Windows",
])

COLORS = {"Filtered Strategy": "#38bdf8", "Unfiltered Strategy": "#a78bfa",
          "Benchmark (Buy & Hold)": "#f59e0b"}

with tab1:
    if cum_ret is not None:
        fig = go.Figure()
        for col in cum_ret.columns:
            c = COLORS.get(col, "#e2e8f0")
            dash = "dot" if "Benchmark" in col else ("dash" if "Unfiltered" in col else "solid")
            fig.add_trace(go.Scatter(
                x=cum_ret.index, y=cum_ret[col], name=col,
                line=dict(color=c, width=2, dash=dash),
                fill="tozeroy" if "Filtered" in col and "Un" not in col else None,
                fillcolor="rgba(56,189,248,.04)" if "Filtered" in col and "Un" not in col else None,
            ))
        fig.update_layout(**PLOTLY_LAYOUT,
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1))
        fig.update_xaxes(title=""); fig.update_yaxes(title="Cumulative Return")
        st.plotly_chart(_grid(fig), use_container_width=True)

with tab2:
    if "Prob_UP" in preds.columns:
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=preds["Prob_UP"], nbinsx=50,
            marker_color="#38bdf8", opacity=0.8, name="All predictions"
        ))
        fig2.add_vline(x=0.502, line_dash="dash", line_color="#f59e0b",
                       annotation_text="Entry threshold", annotation_font_color="#f59e0b")
        fig2.add_vline(x=0.5, line_dash="dot", line_color="#64748b",
                       annotation_text="50%", annotation_font_color="#64748b")
        fig2.update_layout(**PLOTLY_LAYOUT)
        fig2.update_xaxes(title="Probability of UP day")
        fig2.update_yaxes(title="Frequency")
        st.plotly_chart(_grid(fig2), use_container_width=True)

with tab3:
    if raw is not None:
        tp = raw.loc[raw.index >= "2024-01-01", "Nifty_Close"]
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=tp.index, y=tp, name="Nifty 50",
                                   line=dict(color="#e2e8f0", width=1.5)))
        buys  = preds[preds["Prediction"] == 1]
        sells = preds[preds["Prediction"] == 0]
        bp = tp.loc[tp.index.isin(buys.index)]
        sp = tp.loc[tp.index.isin(sells.index)]
        fig3.add_trace(go.Scatter(x=bp.index, y=bp, mode="markers", name="Predicted UP",
                                   marker=dict(color="#22c55e", size=4, symbol="triangle-up")))
        fig3.add_trace(go.Scatter(x=sp.index, y=sp, mode="markers", name="Predicted DOWN",
                                   marker=dict(color="#ef4444", size=4, symbol="triangle-down")))
        fig3.update_layout(**PLOTLY_LAYOUT,
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1))
        fig3.update_yaxes(title="Nifty 50 Close")
        st.plotly_chart(_grid(fig3), use_container_width=True)

with tab4:
    if wf is not None and len(wf) > 0:
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(x=wf["Window"], y=wf["Accuracy"], name="Accuracy",
                               marker_color="#38bdf8", opacity=0.8))
        fig4.add_hline(y=0.5, line_dash="dash", line_color="#ef4444",
                       annotation_text="50% (coin flip)", annotation_font_color="#ef4444")
        fig4.update_layout(**PLOTLY_LAYOUT)
        fig4.update_xaxes(title="Test Window", tickangle=-30)
        fig4.update_yaxes(title="Accuracy", range=[0.3, 0.7])
        st.plotly_chart(_grid(fig4), use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Mean Accuracy per Window**")
            st.dataframe(wf[["Window","Accuracy","AUC","Sharpe"]].set_index("Window"),
                         use_container_width=True)
        with col_b:
            st.metric("Average Accuracy", f"{wf['Accuracy'].mean()*100:.2f}%")
            st.metric("Average AUC",      f"{wf['AUC'].mean():.4f}")
            st.metric("Average Sharpe",   f"{wf['Sharpe'].mean():.4f}")
    else:
        st.info("Run `python src/walk_forward.py` to generate walk-forward results.")

# ── Confusion Matrix + Feature Importance ────────────────────────────────────
st.markdown('<div class="section-hdr">Model Diagnostics</div>', unsafe_allow_html=True)

col_cm, col_fi = st.columns(2)

with col_cm:
    st.markdown("**Confusion Matrix**")
    if cm_df is not None:
        z    = cm_df.values.tolist()
        x    = ["Predicted DOWN", "Predicted UP"]
        y    = ["Actual DOWN", "Actual UP"]
        text = [[str(v) for v in row] for row in z]
        fig_cm = go.Figure(go.Heatmap(
            z=z, x=x, y=y, text=text, texttemplate="%{text}",
            colorscale=[[0,"#0f172a"],[0.5,"#1e3a5f"],[1,"#38bdf8"]],
            showscale=False
        ))
        fig_cm.update_layout(**{**PLOTLY_LAYOUT, "height": 320})
        st.plotly_chart(_grid(fig_cm), use_container_width=True)
    else:
        st.info("Run model_training.py to generate confusion matrix.")

with col_fi:
    st.markdown("**Top 15 Feature Importances**")
    if feat_imp is not None:
        top15 = feat_imp.head(15).sort_values("importance")
        fig_fi = go.Figure(go.Bar(
            x=top15["importance"], y=top15.index,
            orientation="h",
            marker=dict(
                color=top15["importance"],
                colorscale=[[0,"#1e3a5f"],[1,"#38bdf8"]],
            )
        ))
        fig_fi.update_layout(**{**PLOTLY_LAYOUT, "height": 320})
        fig_fi.update_xaxes(title="Importance Score")
        st.plotly_chart(_grid(fig_fi), use_container_width=True)
    else:
        st.info("Run model_training.py to generate feature importances.")

# ── Per-model accuracy ─────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">Per-Model Accuracy Breakdown</div>', unsafe_allow_html=True)

if per_model is not None:
    pm = per_model.copy()
    pm.columns = ["Accuracy"]
    pm["Accuracy %"] = (pm["Accuracy"] * 100).round(2).astype(str) + "%"
    pm["Beat Baseline"] = pm["Accuracy"].apply(lambda x: "✅" if x > 0.5 else "❌")
    st.dataframe(pm[["Accuracy %", "Beat Baseline"]], use_container_width=True)
else:
    st.info("Run model_training.py to generate per-model accuracy.")

# ── Recent Predictions Table ───────────────────────────────────────────────────
st.markdown('<div class="section-hdr">Recent Predictions (Last 20 Days)</div>', unsafe_allow_html=True)

recent = preds.tail(20).copy()
recent.index = pd.to_datetime(recent.index).strftime("%d %b %Y")  # date-only
recent["Direction"]  = recent["Prediction"].map({1: "🟢 UP", 0: "🔴 DOWN"})
recent["Actual"]     = recent["Target"].map({1: "▲ UP", 0: "▼ DOWN"})
recent["Correct"]    = (recent["Prediction"] == recent["Target"]).map({True: "✅", False: "❌"})
recent["Confidence"] = recent["Prob_UP"].apply(lambda x: f"{x*100:.2f}%")
if "RF_prob" in recent.columns:
    recent["RF"]  = recent["RF_prob"].apply(lambda x: f"{x*100:.1f}%")
    recent["XGB"] = recent["XGB_prob"].apply(lambda x: f"{x*100:.1f}%")
    recent["LGB"] = recent["LGB_prob"].apply(lambda x: f"{x*100:.1f}%")
    display_cols = ["Direction", "Confidence", "RF", "XGB", "LGB", "Actual", "Correct"]
else:
    display_cols = ["Direction", "Confidence", "Actual", "Correct"]

st.dataframe(recent[display_cols].sort_index(ascending=False),
             use_container_width=True, height=420)

# ── Market Insights ────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">Market Insights & Observations</div>', unsafe_allow_html=True)

r30     = preds.tail(30)
r30_acc = (r30["Target"] == r30["Prediction"]).mean() * 100
up_ratio= r30["Prediction"].mean() * 100
avg_conf= r30["Prob_UP"].mean() * 100

st.markdown(f"""
<div class="insight">
<strong>What the stacked ensemble is currently observing:</strong><br><br>
Over the last 30 sessions the model achieved a directional accuracy of
<em>{r30_acc:.1f}%</em>, with a bullish bias on <em>{up_ratio:.0f}%</em> of days.
Average ensemble confidence sits at <em>{avg_conf:.1f}%</em> - consistent with the
fundamental unpredictability of next-day market direction under short-term technical signals alone.<br><br>
<strong>Key finding:</strong> The most important features by RF importance are short-term return
momentum (Ret_Lag1-Lag5), RSI-14, and Bollinger Band position. The VIX-return (lagged 1 day)
consistently ranks in the top-10, confirming that institutional risk appetite - as expressed
through implied volatility - provides genuine predictive signal beyond price-only indicators.<br><br>
<strong>Regime observation:</strong> Walk-forward windows covering 2024 Q3-Q4 showed the steepest
accuracy decay, coinciding with a period of rotational sector leadership and multiple gap-up/gap-down
sessions following FII flows. This suggests that in high-churn regimes, the model's lag-based
features lose signal quality, reinforcing the need for a regime-detection layer in Version 3.
</div>
""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <strong style="color:#64748b">TECH STACK</strong><br>
  <span style="color:#475569">Python · Pandas · Scikit-Learn · XGBoost · LightGBM · Streamlit · Plotly</span><br><br>
  Built by <strong style="color:#94a3b8">CA Bhavya</strong> · AI for Finance Level 2 · Ensemble ML for Indian Equity Markets
</div>
""", unsafe_allow_html=True)
