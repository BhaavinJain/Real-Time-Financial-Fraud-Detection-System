# Real-Time Financial Fraud Detection System

> End-to-end ML system for detecting financial fraud on the IEEE-CIS dataset — featuring an XGBoost + Deep Autoencoder ensemble, SHAP explainability, and a live deployed FastAPI backend + Streamlit dashboard.

**🔗 Live Demo:** [Streamlit Dashboard](https://real-time-financial-fraud-detection-system-mimcphs3d5qdtbht66c.streamlit.app/) &nbsp;|&nbsp; **🔌 Live API:** [`/docs`](https://real-time-financial-fraud-detection-2ycq.onrender.com/docs)

> ⏳ Both are hosted on free tiers. The API spins down after ~15 minutes of inactivity — the **first** request after a quiet period can take 30–50s to wake up (visible as "API is waking up" in the dashboard sidebar). Subsequent requests respond in well under a second.

---

## Results

| Metric | Validation | Test (temporal) |
|--------|-----------|-----------------|
| PR-AUC | 0.5919 | 0.4200 |
| F1 Score | 0.5823 | 0.4618 |
| Precision | 0.7603 | 0.6756 |
| Recall | 0.4719 | 0.3508 |
| False Positive Rate | **0.63%** | **0.60%** |

> **0.60% FPR** means only 6 in every 1,000 legitimate transactions are incorrectly blocked — commercially viable for production deployment.

---

## Architecture

```
Raw Transaction (216 features)
         │
         ├──────────────────────────────┐
         │                              │
         ▼                              ▼
  XGBoost Classifier            Deep Autoencoder
  (scale_pos_weight=28.56)      (trained on legit only)
         │                              │
         │  fraud_probability           │  reconstruction_error
         │  [0.0 → 1.0]                 │  normalized to [0.0 → 1.0]
         │                              │
         └──────────────┬───────────────┘
                        │
                        ▼
          ensemble_score = 0.8933 × xgb_proba
                        + 0.1067 × ae_norm_error
                        │
                        ▼
             score ≥ 0.7533 → FRAUD
             score < 0.7533 → LEGITIMATE
                        │
                        ▼
               SHAP explanation
         (top 5 feature contributions)
```

### Deployment architecture

```
┌─────────────────────────┐         ┌──────────────────────────┐
│   Streamlit Dashboard    │  HTTPS  │      FastAPI Backend      │
│  (Streamlit Cloud)       │ ──────► │       (Render)            │
│                          │         │                          │
│  • Batch Review tab      │         │  • POST /predict          │
│  • Quick Demo tab        │         │  • GET  /health           │
│  • Model Performance tab │         │  • GET  /metrics          │
└─────────────────────────┘         └──────────────────────────┘
                                              │
                                              ▼
                                     ┌──────────────────┐
                                     │  models/*.pkl      │
                                     │  autoencoder.keras │
                                     │  shap_explainer    │
                                     └──────────────────┘
```

---

## Project Structure

```
fraud-detection/
├── notebooks/
│   ├── EDA.ipynb                       # Exploratory data analysis
│   ├── feature_engineering.ipynb       # Feature engineering pipeline
│   ├── preprocessing.ipynb             # Splits, encoding, SMOTE, scaling
│   ├── modeling.ipynb                  # XGBoost + LightGBM with Optuna
│   ├── autoencoder.ipynb               # Deep autoencoder for anomaly detection
│   ├── enemble.ipynb                   # Ensemble + final evaluation
│   ├── shap_explainability.ipynb       # SHAP analysis and interpretation
│   ├── card_stats_creation.ipynb       # Per-card amount stats for inference
│   └── generate_sample.py              # Sample transactions for dashboard demo
├── api/                                 # FastAPI application -- LIVE on Render
│   ├── main.py                          # App, routes, model loading (lifespan)
│   ├── schemas.py                       # Pydantic request/response models
│   ├── preprocessing.py                 # Inference-time feature engineering
│   └── requirements.txt
├── dashboard/                            # Streamlit dashboard -- LIVE on Streamlit Cloud
│   ├── app.py                            # Batch Review / Quick Demo / Model Performance
│   ├── sample_transactions.csv           # Demo data (10 fraud + 15 legit, real rows)
│   └── requirements.txt
├── models/                              # Serialized model artifacts
│   ├── xgboost_best.pkl
│   ├── lightgbm_best.pkl
│   ├── autoencoder.keras
│   ├── ensemble_config.pkl
│   ├── shap_explainer.pkl
│   ├── Standard_scaler.pkl
│   ├── target_encoder.pkl
│   ├── label_encoders.pkl
│   ├── card_stats.pkl
│   └── feature_columns.pkl
├── data/
│   ├── raw/                            # IEEE-CIS CSV files (not committed)
│   └── processed/                       # Parquet splits (not committed)
├── reports/
│   ├── shap_summary_plot.png
│   ├── shap_waterfall_*.png
│   ├── xgb_lgb_pr_curve.png
│   ├── ensemble_pr_curve.png
│   └── autoencoder_reconstruction_errors.png
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

**IEEE-CIS Fraud Detection** ([Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection))

- 590,540 transactions, 434 raw features across two tables
- Class imbalance: **3.5% fraud**, 96.5% legitimate
- Two tables merged on `TransactionID`: transaction features + identity features
- Temporal ordering preserved -- data sorted by `TransactionDT` throughout

---

## Methodology

### 1. Feature Engineering

| Feature | Type | Rationale |
|---------|------|-----------|
| `card_id` | Composite | 6 card columns combined into unique card fingerprint |
| `amt_zscore_card` | Behavioral | Is this amount unusual for *this specific card's* history? |
| `amt_log` | Transform | Log of TransactionAmt -- handles right-skewed distribution |
| `hour_sin`, `hour_cos` | Cyclic | Time-of-day without artificial boundary at midnight |
| `card_id_te` | Target encoded | Card-level historical fraud rate (4.3x fraud/clean separation) |
| `P_emaildomain_te` | Target encoded | Email domain fraud rate (protonmail: 40.8% fraud) |
| `ProductCD_te` | Target encoded | Product type fraud rate (Product C: 11.7% fraud) |

**Velocity features** (1h, 24h transaction counts per card) were investigated and dropped after data-driven analysis showed no meaningful fraud/clean separation -- IEEE-CIS fraud is not velocity-driven.

**Missing values:** M-columns -> `'missing'`, other categoricals -> `'unknown'`, numerics -> `-999` (sentinel for tree models).

### 2. Preprocessing Pipeline

```
Full dataset (590,540 rows)
        │
        ▼ Temporal split (not random -- preserves time ordering)
        │
Train (75%) ─── Val (20% of train) ─── Test (25%)
442,905 rows     88,581 rows           147,635 rows
        │
        ▼ Fit on train only, transform all splits
        │
TargetEncoder (P_emaildomain, ProductCD, card_id)
LabelEncoder  (M1, M2, M3, M4, M6)
StandardScaler (for LR and Autoencoder)
        │
        ▼ Train only
        │
SMOTE (for Autoencoder/LR scaled pipeline only)
XGBoost/LightGBM use scale_pos_weight=28.56 on raw imbalanced data
```

**Critical decision:** SMOTE was initially applied for all tree models but caused severe overfitting (train PR-AUC 0.999, val PR-AUC 0.53). Switching to `scale_pos_weight` reduced the overfit gap from 0.467 to 0.349 and FPR from 37% to 0.64%.

### 3. Models

#### XGBoost (Primary Model)
- Hyperparameter tuning: Optuna TPE sampler, 50 trials, PR-AUC objective on validation set
- `scale_pos_weight=28.56` handles class imbalance natively
- Operating threshold: **0.8294** (optimized for F1 on validation)

| Configuration | PR-AUC | FPR |
|---------------|--------|-----|
| Default params | 0.5664 | 6.33% |
| Optuna tuned | 0.5841 | 0.64% |

#### LightGBM
- Same Optuna setup with LightGBM-specific params (num_leaves, min_child_samples)
- PR-AUC: 0.5791, FPR: 0.97% at best threshold
- Smaller overfit gap than XGBoost (0.26 vs 0.35) but lower precision

#### Deep Autoencoder (Anomaly Detection)
- Architecture: `216 -> 128 -> 64 -> 32 -> 64 -> 128 -> 216`
- **Trained exclusively on legitimate transactions** -- learns normal behavior
- Reconstruction error = anomaly score (fraud: 0.0647, legit: 0.0318, **2.03x ratio**)
- Critical fix: clip scaled features to `[-10, 10]` -- raw `-999` sentinels produce `-37,887` after scaling, causing val_loss to explode to 5,000+

#### Ensemble
- Weighted combination: `score = 0.8933 × xgb_proba + 0.1067 × ae_norm_error`
- Weights found by Optuna (100 trials, PR-AUC objective on validation)
- AE errors normalized using percentile clipping (p1-p99) before combining
- **+0.0078 PR-AUC lift** over XGBoost alone with consistent improvement across all metrics

### 4. SHAP Explainability

- **TreeExplainer** (exact Shapley values, not approximate)
- Computed on 5,000 sampled validation transactions
- Base value: 0.04 (model's prior fraud probability)

**Top features by mean |SHAP value|:**

| Rank | Feature | Mean \|SHAP\| | Meaning |
|------|---------|--------------|---------|
| 1 | `card_id` | 1.2079 | Card identity + fraud history -- **4x the next feature** |
| 2 | `C13` | 0.3468 | Vesta count feature |
| 3 | `C1` | 0.2862 | Vesta count feature -- address/card occurrence |
| 4 | `TransactionAmt` | 0.2612 | Raw transaction amount |
| 5 | `C14` | 0.2206 | Vesta count feature -- bidirectional signal |
| 14 | `amt_zscore_card` | 0.1317 | Behavioral anomaly -- amount vs card history |

**Three conceptual feature groups the model learned:**
1. **Card identity & history** (`card_id`, C-columns) -- *who is transacting?*
2. **Behavioral anomaly** (`amt_zscore_card`, `TransactionAmt`) -- *is this unusual?*
3. **Contextual signals** (`P_emaildomain`, D-columns, `addr1`) -- *what is the context?*

**False positive analysis:** A legitimate transaction was flagged with 99.99% confidence because it shared 5 aligned fraud signals (card history, count patterns, Vesta risk features). Only the $43.8 transaction amount argued for legitimacy. This illustrates an inherent limitation of behavioral profiling -- a card with past fraud associations making a genuine purchase.

### 5. Deployment

The system is split into two independently deployed services that communicate over HTTPS:

**FastAPI backend (Render)**
- `POST /predict` -- accepts a raw transaction, runs the full feature engineering + encoding pipeline, returns fraud verdict + ensemble score + top-5 SHAP contributions + plain-English explanation
- `GET /health` -- model version, uptime
- `GET /metrics` -- in-memory running totals (predictions served, fraud rate, average scores)
- All models loaded once at startup via a `lifespan` context manager -- no per-request reloading
- All file paths resolved relative to `__file__`, not the working directory, so the service runs identically locally and on Render
- Inference-time feature engineering exactly mirrors training: builds `card_id` from raw card fields, computes `amt_log`/`amt_zscore_card`/`hour_sin`/`hour_cos`, applies the *same fitted* `TargetEncoder` and `LabelEncoder`s saved from training (never refit)

**Streamlit dashboard (Streamlit Community Cloud)**
- **Batch Review** -- upload a CSV or load a 25-row sample; scores the whole batch through the live API and surfaces flagged transactions sorted by risk, with full SHAP drill-down per row. This reflects how the system is actually used in production -- a fraud analyst triages a list, never types in 216 fields by hand.
- **Quick Demo** -- a simplified ~8-field form (amount, product, card network/type, email domain) for live demonstration. Every other feature defaults through the same imputation logic used in training. A visible note clarifies that in production, 200+ additional signals are captured automatically by the payment processor.
- **Model Performance** -- training-time metrics and SHAP feature importance, for context without needing to open a notebook.

---

## Key Engineering Decisions

| Decision | Naive Approach | What We Did | Why |
|----------|---------------|-------------|-----|
| Train-test split | `random_state=42` shuffle | Temporal 75/25 split | Fraud patterns evolve over time -- shuffling creates leakage |
| Class imbalance (trees) | SMOTE | `scale_pos_weight=28.56` | SMOTE caused train PR-AUC 0.999 / val 0.53 -- memorized synthetic patterns |
| Class imbalance (AE) | SMOTE | Raw legit transactions | Autoencoder must learn real normal patterns, not synthetic ones |
| Card identity | `card1` alone | Composite `card_id` (6 cols) | `card1` alone merges different cards -- investigated and confirmed fragmentation |
| Target encoding | Fit on full data | Fit on train fold only | Fitting on full data leaks test fraud rates into training |
| AE sentinel handling | No clipping | Clip to `[-10, 10]` | `-999` -> `-37,887` after scaling, explodes MSE loss |
| AE normalization | Min-max | Percentile (p1-p99) | Min-max compresses signal to std=0.014; percentile gives std=0.128 |
| SHAP explainer | KernelExplainer | TreeExplainer | Exact values (not approximate), 100x faster for tree models |
| Model loading paths | Relative (`../models/...`) | Absolute, via `Path(__file__)` | Relative paths broke on Render -- resolved differently depending on launch directory |
| Feature columns at startup | Load full 189MB training parquet | Small `feature_columns.pkl` artifact | Reduced both startup time and deploy footprint |
| Python runtime | Platform default | Pinned `3.11` (Render + Streamlit Cloud) | Newer defaults (3.13/3.14) lacked prebuilt wheels for pandas/numpy, forcing slow source builds and version conflicts |

---

## Reports

### Precision-Recall Curves

XGBoost vs LightGBM vs Ensemble -- precision stays above 0.90 until recall hits ~0.30, indicating high-confidence predictions are very reliable.

### SHAP Summary Plot

Beeswarm plot across 5,000 transactions -- `card_id` shows the widest spread (-4 to +4), confirming it as the primary discriminator. Protonmail transactions visible as outlier red dots in `P_emaildomain`.

### Reconstruction Error Distribution

Fraud transactions show a long tail extending to high reconstruction errors while legitimate transactions cluster tightly at low errors -- confirms the autoencoder learned a genuine anomaly signal.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Data processing | pandas, numpy |
| Feature encoding | scikit-learn TargetEncoder, LabelEncoder |
| Class balancing | imbalanced-learn SMOTE, XGBoost scale_pos_weight |
| Hyperparameter tuning | Optuna (TPE sampler) |
| Tree models | XGBoost, LightGBM |
| Deep learning | TensorFlow / Keras |
| Explainability | SHAP (TreeExplainer) |
| API | FastAPI + Uvicorn -- **deployed on Render** |
| Dashboard | Streamlit + Plotly -- **deployed on Streamlit Community Cloud** |

---

## Setup (local development)

```bash
# Clone the repo
git clone https://github.com/BhaavinJain/Real-Time-Financial-Fraud-Detection-System.git
cd Real-Time-Financial-Fraud-Detection-System

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle
# Place train_transaction.csv and train_identity.csv in data/raw/

# Run notebooks in order (EDA -> feature engineering -> preprocessing ->
# modeling -> autoencoder -> ensemble -> shap_explainability)
jupyter notebook notebooks/EDA.ipynb

# Run the API locally
cd api
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Run the dashboard locally (separate terminal)
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```


## License

MIT
