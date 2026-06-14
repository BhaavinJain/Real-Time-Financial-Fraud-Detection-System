# Real-Time Financial Fraud Detection System

> End-to-end ML system for detecting financial fraud on the IEEE-CIS dataset — featuring XGBoost + Deep Autoencoder ensemble, SHAP explainability, FastAPI deployment, Streamlit dashboard, and Evidently AI drift monitoring.

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

---

## Project Structure

```
fraud-detection/
├── notebooks/
│   ├── 01_EDA.ipynb                    # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb    # Feature engineering pipeline
│   ├── 03_preprocessing.ipynb          # Splits, encoding, SMOTE, scaling
│   ├── 04_05_modeling.ipynb            # XGBoost + LightGBM with Optuna
│   ├── 05_autoencoder.ipynb            # Deep autoencoder for anomaly detection
│   └── 06_ensemble.ipynb               # Ensemble + final evaluation
│   └── 07_shap_explainability.ipynb    # SHAP analysis and interpretation
├── src/                                # Source modules (WIP)
├── api/                                # FastAPI application (WIP)
├── dashboard/                          # Streamlit dashboard (WIP)
├── docker/                             # Dockerfiles (WIP)
├── models/                             # Serialized model artifacts
│   ├── xgboost_best.pkl
│   ├── lightgbm_best.pkl
│   ├── autoencoder.keras
│   ├── ensemble_config.pkl
│   ├── shap_explainer.pkl
│   ├── Standard_scaler.pkl
│   ├── target_encoder.pkl
│   └── label_encoders.pkl
├── data/
│   ├── raw/                            # IEEE-CIS CSV files (not committed)
│   └── processed/                      # Parquet splits (not committed)
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
- Temporal ordering preserved — data sorted by `TransactionDT` throughout

---

## Methodology

### 1. Feature Engineering

| Feature | Type | Rationale |
|---------|------|-----------|
| `card_id` | Composite | 6 card columns combined into unique card fingerprint |
| `amt_zscore_card` | Behavioral | Is this amount unusual for *this specific card's* history? |
| `amt_log` | Transform | Log of TransactionAmt — handles right-skewed distribution |
| `hour_sin`, `hour_cos` | Cyclic | Time-of-day without artificial boundary at midnight |
| `card_id_te` | Target encoded | Card-level historical fraud rate (4.3x fraud/clean separation) |
| `P_emaildomain_te` | Target encoded | Email domain fraud rate (protonmail: 40.8% fraud) |
| `ProductCD_te` | Target encoded | Product type fraud rate (Product C: 11.7% fraud) |

**Velocity features** (1h, 24h transaction counts per card) were investigated and dropped after data-driven analysis showed no meaningful fraud/clean separation — IEEE-CIS fraud is not velocity-driven.

**Missing values:** M-columns → `'missing'`, other categoricals → `'unknown'`, numerics → `-999` (sentinel for tree models).

### 2. Preprocessing Pipeline

```
Full dataset (590,540 rows)
        │
        ▼ Temporal split (not random — preserves time ordering)
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
- Architecture: `216 → 128 → 64 → 32 → 64 → 128 → 216`
- **Trained exclusively on legitimate transactions** — learns normal behavior
- Reconstruction error = anomaly score (fraud: 0.0647, legit: 0.0318, **2.03x ratio**)
- Critical fix: clip scaled features to `[-10, 10]` — raw `-999` sentinels produce `-37,887` after scaling, causing val_loss to explode to 5,000+

#### Ensemble
- Weighted combination: `score = 0.8933 × xgb_proba + 0.1067 × ae_norm_error`
- Weights found by Optuna (100 trials, PR-AUC objective on validation)
- AE errors normalized using percentile clipping (p1–p99) before combining
- **+0.0078 PR-AUC lift** over XGBoost alone with consistent improvement across all metrics

### 4. SHAP Explainability

- **TreeExplainer** (exact Shapley values, not approximate)
- Computed on 5,000 sampled validation transactions
- Base value: 0.04 (model's prior fraud probability)

**Top features by mean |SHAP value|:**

| Rank | Feature | Mean \|SHAP\| | Meaning |
|------|---------|--------------|---------|
| 1 | `card_id` | 1.2079 | Card identity + fraud history — **4x the next feature** |
| 2 | `C13` | 0.3468 | Vesta count feature |
| 3 | `C1` | 0.2862 | Vesta count feature — address/card occurrence |
| 4 | `TransactionAmt` | 0.2612 | Raw transaction amount |
| 5 | `C14` | 0.2206 | Vesta count feature — bidirectional signal |
| 14 | `amt_zscore_card` | 0.1317 | Behavioral anomaly — amount vs card history |

**Three conceptual feature groups the model learned:**
1. **Card identity & history** (`card_id`, C-columns) — *who is transacting?*
2. **Behavioral anomaly** (`amt_zscore_card`, `TransactionAmt`) — *is this unusual?*
3. **Contextual signals** (`P_emaildomain`, D-columns, `addr1`) — *what is the context?*

**False positive analysis:** A legitimate transaction was flagged with 99.99% confidence because it shared 5 aligned fraud signals (card history, count patterns, Vesta risk features). Only the $43.8 transaction amount argued for legitimacy. This illustrates an inherent limitation of behavioral profiling — a card with past fraud associations making a genuine purchase.

---

## Key Engineering Decisions

| Decision | Naive Approach | What We Did | Why |
|----------|---------------|-------------|-----|
| Train-test split | `random_state=42` shuffle | Temporal 75/25 split | Fraud patterns evolve over time — shuffling creates leakage |
| Class imbalance (trees) | SMOTE | `scale_pos_weight=28.56` | SMOTE caused train PR-AUC 0.999 / val 0.53 — memorized synthetic patterns |
| Class imbalance (AE) | SMOTE | Raw legit transactions | Autoencoder must learn real normal patterns, not synthetic ones |
| Card identity | `card1` alone | Composite `card_id` (6 cols) | `card1` alone merges different cards — investigated and confirmed fragmentation |
| Target encoding | Fit on full data | Fit on train fold only | Fitting on full data leaks test fraud rates into training |
| AE sentinel handling | No clipping | Clip to `[-10, 10]` | `-999` → `-37,887` after scaling, explodes MSE loss |
| AE normalization | Min-max | Percentile (p1-p99) | Min-max compresses signal to std=0.014; percentile gives std=0.128 |
| SHAP explainer | KernelExplainer | TreeExplainer | Exact values (not approximate), 100x faster for tree models |

---

## Reports

### Precision-Recall Curves

XGBoost vs LightGBM vs Ensemble — precision stays above 0.90 until recall hits ~0.30, indicating high-confidence predictions are very reliable.

### SHAP Summary Plot

Beeswarm plot across 5,000 transactions — `card_id` shows the widest spread (-4 to +4), confirming it as the primary discriminator. Protonmail transactions visible as outlier red dots in `P_emaildomain`.

### Reconstruction Error Distribution

Fraud transactions show a long tail extending to high reconstruction errors while legitimate transactions cluster tightly at low errors — confirms the autoencoder learned a genuine anomaly signal.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Data processing | pandas, numpy |
| Feature encoding | scikit-learn TargetEncoder, LabelEncoder |
| Class balancing | imbalanced-learn SMOTE, XGBoost scale_pos_weight |
| Hyperparameter tuning | Optuna (TPE sampler, 50 trials each) |
| Tree models | XGBoost, LightGBM |
| Deep learning | TensorFlow / Keras |
| Explainability | SHAP (TreeExplainer) |
| API | FastAPI + Uvicorn *(in progress)* |
| Dashboard | Streamlit *(in progress)* |
| Drift monitoring | Evidently AI *(in progress)* |
| Containerization | Docker + docker-compose *(in progress)* |
| Deployment | Render (API) + HuggingFace Spaces (dashboard) *(in progress)* |

---

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/fraud-detection.git
cd fraud-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle
# Place train_transaction.csv and train_identity.csv in data/raw/

# Run notebooks in order
jupyter notebook notebooks/01_EDA.ipynb
```

---

## What's Next

- [ ] FastAPI REST API (`POST /predict` with SHAP output, `GET /health`, `GET /metrics`)
- [ ] Streamlit dashboard with real-time predictions and SHAP visualization
- [ ] Docker + docker-compose for full containerized deployment
- [ ] Render + HuggingFace Spaces live deployment
- [ ] Evidently AI drift monitoring report
- [ ] Optional: Natural language fraud explanation via Ollama/Llama3