"""
Fraud Detection Dashboard
Run with: streamlit run app.py

Three tabs:
  1. Batch Review   -- realistic fraud-analyst workflow: triage a list of
                        already-scored transactions, drill into SHAP per row
  2. Quick Demo     -- simplified single-transaction form for live demos
                        (only the handful of fields a human would reasonably
                        know; everything else defaults via the API schema)
  3. Model Performance -- PR curves, SHAP summary, key metrics from training
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection — Analyst Console",
    page_icon="\U0001F6E1",
    layout="wide",
    initial_sidebar_state="expanded",
)

import os

# st.secrets always exists as an attribute even with no secrets.toml file --
# hasattr() can't detect that, so we must try/except the actual access.
try:
    API_URL = st.secrets.get("API_URL", "https://real-time-financial-fraud-detection-2ycq.onrender.com/")
except Exception:
    API_URL = "https://real-time-financial-fraud-detection-2ycq.onrender.com/"

API_URL = os.environ.get("API_URL", API_URL)

# ── Design tokens (dark theme) ──────────────────────────────────────────────
PAGE_BG      = "#0B0F14"   # near-black app background
SURFACE      = "#12171F"   # cards, tabs, inputs
SURFACE_HOVER = "#1A212B"
NAVY         = "#0B1B2E"   # header gradient start (kept dark-navy, not black, for depth)
NAVY_LIGHT   = "#15293F"
ACCENT       = "#5BA3E0"   # brighter blue for dark backgrounds
ACCENT_DARK  = "#3B8BD4"
RISK_RED     = "#FF6B4A"
RISK_RED_BG  = "#2A1812"
SAFE_GREEN   = "#3DDC97"
SAFE_GREEN_BG = "#0F2A21"
AMBER        = "#E8A33D"
AMBER_BG     = "#2A2112"
TEXT_PRIMARY = "#E8EDF4"
TEXT_MUTE    = "#8A99AC"
BORDER       = "#26303D"
CARD_BG      = "#12171F"

CUSTOM_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'IBM Plex Sans', sans-serif;
        color: {TEXT_PRIMARY};
    }}

    .stApp {{
        background-color: {PAGE_BG};
    }}

    /* Default text color across markdown, headers, body copy */
    .stMarkdown, .stMarkdown p, .stMarkdown li,
    h1, h2, h3, h4, h5, h6, label, .stCaption {{
        color: {TEXT_PRIMARY} !important;
    }}

    section[data-testid="stSidebar"] {{
        background-color: {NAVY};
        border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] * {{
        color: #C3D2E3 !important;
    }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: #FFFFFF !important;
    }}

    .app-header {{
        background: linear-gradient(135deg, {NAVY} 0%, {NAVY_LIGHT} 100%);
        padding: 28px 36px;
        border-radius: 14px;
        margin-bottom: 24px;
        color: white;
        border: 1px solid {BORDER};
    }}
    .app-header .eyebrow {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: {ACCENT};
        margin-bottom: 6px;
    }}
    .app-header h1 {{
        font-size: 26px;
        font-weight: 700;
        margin: 0 0 4px 0;
        color: white !important;
    }}
    .app-header p {{
        font-size: 14px;
        color: #9FB3C8 !important;
        margin: 0;
    }}

    .metric-card {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 16px 18px;
    }}
    .metric-label {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {TEXT_MUTE} !important;
        margin-bottom: 4px;
    }}
    .metric-value {{
        font-size: 26px;
        font-weight: 700;
        color: {TEXT_PRIMARY} !important;
    }}

    .verdict-fraud {{
        background: {RISK_RED_BG};
        border: 1px solid {RISK_RED};
        border-radius: 10px;
        padding: 20px;
    }}
    .verdict-safe {{
        background: {SAFE_GREEN_BG};
        border: 1px solid {SAFE_GREEN};
        border-radius: 10px;
        padding: 20px;
    }}
    .verdict-title-fraud {{
        color: {RISK_RED} !important;
        font-size: 22px;
        font-weight: 700;
        margin: 0;
    }}
    .verdict-title-safe {{
        color: {SAFE_GREEN} !important;
        font-size: 22px;
        font-weight: 700;
        margin: 0;
    }}

    .note-box {{
        background: {AMBER_BG};
        border-left: 3px solid {AMBER};
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 13px;
        color: #E8C98A !important;
    }}

    div[data-testid="stMetricValue"] {{
        font-size: 24px;
        color: {TEXT_PRIMARY} !important;
    }}
    div[data-testid="stMetricLabel"] {{
        color: {TEXT_MUTE} !important;
    }}

    /* ── Tabs: explicit colors for every state so nothing depends on
       browser/theme defaults or hover-only visibility ── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        border-bottom: 1px solid {BORDER};
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        border-radius: 8px 8px 0 0;
        padding: 10px 18px;
        font-weight: 600;
        color: {TEXT_MUTE} !important;
        transition: color 0.15s ease, background-color 0.15s ease;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: {SURFACE_HOVER};
        color: {TEXT_PRIMARY} !important;
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        color: {ACCENT} !important;
        background-color: {SURFACE};
    }}
    .stTabs [data-baseweb="tab"] p {{
        color: inherit !important;
    }}
    .stTabs [data-baseweb="tab-highlight"] {{
        background-color: {ACCENT};
    }}

    /* Inputs, selects, file uploader -- match dark surface */
    div[data-testid="stFileUploader"],
    div[data-baseweb="select"] > div,
    .stNumberInput input, .stTextInput input {{
        background-color: {SURFACE} !important;
        color: {TEXT_PRIMARY} !important;
        border-color: {BORDER} !important;
    }}

    /* DataFrame / table */
    div[data-testid="stDataFrame"] {{
        background-color: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 8px;
    }}

    /* Buttons */
    .stButton button {{
        background-color: {SURFACE};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
    }}
    .stButton button:hover {{
        background-color: {SURFACE_HOVER};
        border-color: {ACCENT};
        color: {ACCENT} !important;
    }}
    .stButton button[kind="primary"] {{
        background-color: {ACCENT_DARK};
        color: white;
        border: none;
    }}
    .stButton button[kind="primary"]:hover {{
        background-color: {ACCENT};
        color: white !important;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="app-header">
    <div class="eyebrow">IEEE-CIS &middot; XGBoost + Autoencoder Ensemble</div>
    <h1>Fraud Detection — Analyst Console</h1>
    <p>Real-time transaction scoring with SHAP-backed explanations. Connected to {API_URL}</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### System status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success(f"API online &middot; v{health.get('model_version', '—')}", icon="\u2705")
        st.caption(f"Uptime: {health.get('uptime_seconds', 0):.0f}s")
    except Exception:
        st.error("API unreachable — start the FastAPI server", icon="\u26A0\uFE0F")
        st.code("cd api && uvicorn main:app --reload", language="bash")

    st.markdown("---")
    st.markdown("### Running metrics")
    try:
        m = requests.get(f"{API_URL}/metrics", timeout=3).json()
        st.metric("Predictions served", m.get("total_predictions", 0))
        st.metric("Fraud flagged", m.get("fraud_flagged", 0))
        st.metric("Fraud rate", f"{m.get('fraud_rate', 0) * 100:.2f}%")
    except Exception:
        st.caption("No metrics available")

    st.markdown("---")
    st.markdown("### About this system")
    st.caption(
        "This dashboard mirrors how fraud detection actually operates in "
        "production: transactions are scored automatically by a payment "
        "processor, not typed in by hand. The **Quick Demo** tab simplifies "
        "input for illustration; the **Batch Review** tab reflects the real "
        "analyst workflow."
    )


# ── Helper: call the API ─────────────────────────────────────────────────
def call_predict(payload: dict) -> dict:
    resp = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
    resp.raise_for_status()
    return resp.json()


def render_verdict(result: dict):
    is_fraud = result["fraud"] == 1
    css_class = "verdict-fraud" if is_fraud else "verdict-safe"
    title_class = "verdict-title-fraud" if is_fraud else "verdict-title-safe"
    label = "FRAUD FLAGGED" if is_fraud else "TRANSACTION CLEARED"
    icon = "\U0001F6A8" if is_fraud else "\u2705"

    st.markdown(f"""
    <div class="{css_class}">
        <p class="{title_class}">{icon} {label}</p>
        <p style="margin:6px 0 0 0; color:#3A4654; font-size:14px;">{result['explanation']}</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Ensemble score", f"{result['ensemble_score']:.4f}",
                   help="Weighted combination of XGBoost probability and autoencoder anomaly signal")
    with c2:
        st.metric("XGBoost probability", f"{result['xgb_probability']:.4f}")
    with c3:
        st.metric("Decision threshold", f"{result['threshold']:.4f}")

    st.markdown("##### Why this decision — top contributing features")
    shap_df = pd.DataFrame(
        [{"feature": k, "shap_value": v} for k, v in result["shap_top5"].items()]
    ).sort_values("shap_value")

    fig = go.Figure(go.Bar(
        x=shap_df["shap_value"],
        y=shap_df["feature"],
        orientation="h",
        marker_color=[RISK_RED if v > 0 else SAFE_GREEN for v in shap_df["shap_value"]],
        text=[f"{v:+.3f}" for v in shap_df["shap_value"]],
        textposition="outside",
    ))
    fig.update_layout(
        height=240,
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis_title="SHAP value (\u2190 reduces risk &nbsp;|&nbsp; increases risk \u2192)",
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        font=dict(family="IBM Plex Sans", size=12, color=TEXT_PRIMARY),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, color=TEXT_MUTE),
        yaxis=dict(color=TEXT_PRIMARY),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Tabs ──────────────────────────────────────────────────────────────────
tab_batch, tab_demo, tab_perf = st.tabs([
    "\U0001F4CB  Batch Review",
    "\u26A1  Quick Demo",
    "\U0001F4CA  Model Performance",
])


# ═══════════════════════════════════════════════════════════════════════
# TAB 1 — BATCH REVIEW (the realistic analyst workflow)
# ═══════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("""
    <div class="note-box">
    This is how the system is actually used in production — a fraud analyst
    doesn't type in transactions. The payment processor sends transaction
    data automatically, the model scores all of them, and the analyst
    reviews a prioritized list of what was flagged.
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    uploaded = st.file_uploader(
        "Upload a transaction batch (CSV with raw transaction columns)",
        type=["csv"],
        help="Expects columns matching the API's TransactionInput schema"
    )

    sample_path = Path(__file__).parent / "sample_transactions.csv"
    use_sample = st.button("Or load 25 sample transactions from test set", type="secondary")

    batch_df = None
    sample_missing = False
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
    elif use_sample and sample_path.exists():
        batch_df = pd.read_csv(sample_path)
    elif use_sample:
        sample_missing = True

    if sample_missing:
        st.markdown("""
        <div style="background:#FDECEA; border-left:3px solid #D85A30;
                    border-radius:6px; padding:14px 18px; margin:8px 0;
                    color:#7A2E1A; font-size:14px;">
            <strong>sample_transactions.csv not found.</strong> Run
            <code style="background:#FFF3F0; padding:2px 6px; border-radius:4px;">
            generate_sample_transactions.py</code> from your <code style="background:#FFF3F0; padding:2px 6px; border-radius:4px;">notebooks/</code>
            folder first — see setup note below.
        </div>
        """, unsafe_allow_html=True)

    if batch_df is not None:
        has_ground_truth = "_true_label" in batch_df.columns
        st.write(f"Loaded **{len(batch_df)}** transactions. Scoring...")
        progress = st.progress(0)
        results = []

        for i, row in batch_df.iterrows():
            # Strip reference/debug columns (prefixed with _) before sending
            # to the API -- these are for demo validation only, never part
            # of the real schema.
            payload = {k: v for k, v in row.dropna().to_dict().items() if not k.startswith('_')}
            try:
                res = call_predict(payload)
                results.append({
                    "row": i,
                    "TransactionAmt": row.get("TransactionAmt"),
                    "ProductCD": row.get("ProductCD"),
                    "fraud": res["fraud"],
                    "true_label": int(row["_true_label"]) if has_ground_truth else None,
                    "ensemble_score": res["ensemble_score"],
                    "top_reason": list(res["shap_top5"].keys())[0],
                    "explanation": res["explanation"],
                    "_raw": res,
                })
            except Exception as e:
                results.append({
                    "row": i, "TransactionAmt": row.get("TransactionAmt"),
                    "ProductCD": row.get("ProductCD"), "fraud": None,
                    "true_label": int(row["_true_label"]) if has_ground_truth else None,
                    "ensemble_score": None, "top_reason": "ERROR",
                    "explanation": str(e), "_raw": None,
                })
            progress.progress((i + 1) / len(batch_df))

        results_df = pd.DataFrame(results)
        flagged = results_df[results_df["fraud"] == 1]

        c1, c2, c3, c4 = st.columns(4) if has_ground_truth else (st.columns(3) + [None])
        c1.metric("Transactions scored", len(results_df))
        c2.metric("Flagged as fraud", len(flagged))
        c3.metric("Flag rate", f"{len(flagged) / len(results_df) * 100:.1f}%" if len(results_df) else "—")
        if has_ground_truth and c4 is not None:
            correct = (results_df["fraud"] == results_df["true_label"]).sum()
            c4.metric("Matches ground truth", f"{correct}/{len(results_df)}",
                      help="Only available for demo data with known labels -- not available in production")

        st.markdown("##### Flagged transactions — sorted by risk")
        if len(flagged):
            display_cols = ["row", "TransactionAmt", "ProductCD", "ensemble_score", "top_reason"]
            if has_ground_truth:
                display_cols.insert(1, "true_label")
            sorted_flagged = flagged.sort_values("ensemble_score", ascending=False)
            st.dataframe(
                sorted_flagged[display_cols].style.format({"ensemble_score": "{:.4f}"}),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("##### Drill into a flagged transaction")
            selected_row = st.selectbox(
                "Select a row to inspect",
                sorted_flagged["row"].tolist(),
                format_func=lambda r: f"Row {r} — amount ${results_df.loc[results_df['row']==r, 'TransactionAmt'].values[0]:.2f}"
            )
            selected_result = results_df.loc[results_df["row"] == selected_row, "_raw"].values[0]
            if selected_result:
                render_verdict(selected_result)
        else:
            st.info("No transactions in this batch were flagged as fraud.")
    else:
        st.markdown("""
        <div style="font-size:13px; color:#5B6B7F; padding:8px 0;">
            No batch loaded yet. Upload a CSV or click the sample button above.<br>
            <strong>Setup note:</strong> run
            <code style="background:#EEF2F6; padding:2px 6px; border-radius:4px; color:#185FA5;">generate_sample_transactions.py</code>
            from <code style="background:#EEF2F6; padding:2px 6px; border-radius:4px; color:#185FA5;">notebooks/</code>
            once to create
            <code style="background:#EEF2F6; padding:2px 6px; border-radius:4px; color:#185FA5;">sample_transactions.csv</code>
            next to this file.
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB 2 — QUICK DEMO (simplified single-transaction form)
# ═══════════════════════════════════════════════════════════════════════
with tab_demo:
    st.markdown("""
    <div class="note-box">
    For live demonstration only. In production, every field below — plus
    ~200 more (device fingerprint, address history, network risk scores) —
    is captured automatically by the payment processor in milliseconds.
    This form exposes only the handful of signals a person would reasonably
    recognize; everything else defaults via the API's preprocessing pipeline.
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    with st.form("quick_demo_form"):
        col1, col2 = st.columns(2)

        with col1:
            amount = st.number_input("Transaction amount ($)", min_value=0.01, value=150.0, step=10.0)
            product = st.selectbox("Product type", ["W", "C", "H", "R", "S"],
                                     help="Product C historically carries the highest fraud rate (11.7%)")
            card_network = st.selectbox("Card network", ["visa", "mastercard", "discover", "american express"])
            card_type = st.radio("Card type", ["debit", "credit"], horizontal=True)

        with col2:
            email_domain = st.selectbox(
                "Purchaser email domain",
                ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
                 "aol.com", "icloud.com", "protonmail.com", "mail.com", "unknown"],
                help="protonmail.com and mail.com carry elevated fraud rates in this dataset"
            )
            card_id_num = st.number_input("Card identifier (card1)", min_value=1000, max_value=20000, value=13926)
            hour_of_day = st.slider("Hour of day", 0, 23, 14)
            day_offset = st.number_input("Days since dataset start", min_value=0, value=1)

        submitted = st.form_submit_button("Check transaction", type="primary", use_container_width=True)

    if submitted:
        # Map the simplified inputs to the raw fields the API expects.
        # Everything not listed here falls back to schema defaults
        # (-999.0 / "unknown" / "missing"), exactly matching training imputation.
        transaction_dt = day_offset * 86400 + hour_of_day * 3600

        payload = {
            "TransactionAmt": amount,
            "TransactionDT": transaction_dt,
            "ProductCD": product,
            "card1": int(card_id_num),
            "card4": card_network,
            "card6": card_type,
            "P_emaildomain": email_domain,
        }

        with st.spinner("Scoring transaction..."):
            try:
                result = call_predict(payload)
                st.write("")
                render_verdict(result)
            except requests.exceptions.ConnectionError:
                st.error("Could not reach the API. Confirm FastAPI is running on " + API_URL)
            except Exception as e:
                st.error(f"Prediction failed: {e}")


# ═══════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE (training-time results, for context)
# ═══════════════════════════════════════════════════════════════════════
with tab_perf:
    st.markdown("##### Validation & test performance")

    perf_data = pd.DataFrame([
        {"Model": "XGBoost (Optuna)", "Split": "Validation", "PR-AUC": 0.5841, "F1": 0.5814, "Precision": 0.7557, "Recall": 0.4724, "FPR": 0.0064},
        {"Model": "LightGBM (Optuna)", "Split": "Validation", "PR-AUC": 0.5791, "F1": 0.5650, "Precision": 0.6776, "Recall": 0.4845, "FPR": 0.0097},
        {"Model": "Autoencoder", "Split": "Validation", "PR-AUC": 0.0982, "F1": 0.1948, "Precision": 0.1425, "Recall": 0.3074, "FPR": 0.0778},
        {"Model": "Ensemble (deployed)", "Split": "Validation", "PR-AUC": 0.5919, "F1": 0.5823, "Precision": 0.7603, "Recall": 0.4719, "FPR": 0.0063},
        {"Model": "Ensemble (deployed)", "Split": "Test", "PR-AUC": 0.4200, "F1": 0.4618, "Precision": 0.6756, "Recall": 0.3508, "FPR": 0.0060},
    ])

    c1, c2, c3, c4 = st.columns(4)
    deployed_val = perf_data[(perf_data.Model == "Ensemble (deployed)") & (perf_data.Split == "Validation")].iloc[0]
    c1.markdown(f'<div class="metric-card"><div class="metric-label">PR-AUC (val)</div><div class="metric-value">{deployed_val["PR-AUC"]:.4f}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-label">Precision</div><div class="metric-value">{deployed_val["Precision"]:.2%}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-label">Recall</div><div class="metric-value">{deployed_val["Recall"]:.2%}</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-label">False positive rate</div><div class="metric-value">{deployed_val["FPR"]:.2%}</div></div>', unsafe_allow_html=True)

    st.write("")
    st.dataframe(
        perf_data.style.format({
            "PR-AUC": "{:.4f}", "F1": "{:.4f}", "Precision": "{:.4f}",
            "Recall": "{:.4f}", "FPR": "{:.2%}"
        }),
        use_container_width=True, hide_index=True
    )

    st.markdown("""
    <div class="note-box">
    Note on the validation \u2192 test gap: PR-AUC drops from 0.59 to 0.42 due to
    temporal drift — the test set contains more recent transactions with fraud
    patterns the model saw less of during training. The deployed threshold
    (0.7533) was chosen to prioritize precision: the model misses roughly half
    of all fraud (recall ~47%) in exchange for a false-positive rate under 1%,
    meaning fewer legitimate customers are inconvenienced.
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.markdown("##### Top features by SHAP importance")

    shap_importance = pd.DataFrame([
        {"feature": "card_id", "mean_abs_shap": 1.2079},
        {"feature": "C13", "mean_abs_shap": 0.3468},
        {"feature": "C1", "mean_abs_shap": 0.2862},
        {"feature": "TransactionAmt", "mean_abs_shap": 0.2612},
        {"feature": "C14", "mean_abs_shap": 0.2206},
        {"feature": "P_emaildomain", "mean_abs_shap": 0.1976},
        {"feature": "V70", "mean_abs_shap": 0.1788},
        {"feature": "D15", "mean_abs_shap": 0.1731},
        {"feature": "D2", "mean_abs_shap": 0.1684},
        {"feature": "addr1", "mean_abs_shap": 0.1640},
    ]).sort_values("mean_abs_shap")

    fig2 = go.Figure(go.Bar(
        x=shap_importance["mean_abs_shap"],
        y=shap_importance["feature"],
        orientation="h",
        marker_color=ACCENT,
    ))
    fig2.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Mean |SHAP value| across 5,000 validation transactions",
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        font=dict(family="IBM Plex Sans", size=12, color=TEXT_PRIMARY),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, color=TEXT_MUTE),
        yaxis=dict(color=TEXT_PRIMARY),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        "card_id dominates with 4x the impact of the next feature, confirming "
        "that card-level historical behavior is the model's primary signal — "
        "consistent with how production fraud systems weight identity history."
    )