"""
PATCH for api/main.py — replace the existing imports + lifespan function
with this version. Everything else in main.py (the /health, /metrics,
/predict route handlers) stays exactly the same — only the path resolution
at startup changes.
"""

import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow import keras

from schemas import TransactionInput, PredictionResponse, HealthResponse, MetricsResponse
from preprocessing import engineer_features, apply_encoders, align_columns

# ── Path resolution -- ABSOLUTE, based on this file's own location ─────────
# __file__ = .../api/main.py  ->  parent = api/  ->  parent.parent = repo root
# This works identically on your local machine, Render, Docker, anywhere --
# it never depends on what directory the process happened to be launched from.
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

# ── Global state -- populated once at startup, never reloaded per-request ──
ml_models = {}
app_state = {
    'start_time': time.time(),
    'total_predictions': 0,
    'fraud_flagged': 0,
    'xgb_proba_sum': 0.0,
    'ensemble_score_sum': 0.0,
    'last_prediction_time': None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ──
    print(f"BASE_DIR resolved to: {BASE_DIR}")
    print(f"Loading models from: {MODELS_DIR}")

    ml_models['xgb']             = joblib.load(MODELS_DIR / 'xgboost_best.pkl')
    ml_models['autoencoder']     = keras.models.load_model(MODELS_DIR / 'autoencoder.keras')
    ml_models['scaler']          = joblib.load(MODELS_DIR / 'Standard_scaler.pkl')
    ml_models['target_encoder']  = joblib.load(MODELS_DIR / 'target_encoder.pkl')
    ml_models['label_encoders']  = joblib.load(MODELS_DIR / 'label_encoders.pkl')
    ml_models['shap_explainer']  = joblib.load(MODELS_DIR / 'shap_explainer.pkl')
    ml_models['ensemble_config'] = joblib.load(MODELS_DIR / 'ensemble_config.pkl')
    ml_models['card_stats']      = joblib.load(MODELS_DIR / 'card_stats.pkl')

    ml_models['feature_columns'] = joblib.load(MODELS_DIR / 'feature_columns.pkl')
    ml_models['te_cols'] = ['P_emaildomain', 'ProductCD', 'card_id']

    print(f"Models loaded. {len(ml_models['feature_columns'])} features expected.")
    print(f"card_stats loaded for {len(ml_models['card_stats']):,} known cards.")
    yield
    # ── SHUTDOWN ──
    print("Shutting down API")
 
 
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time transaction fraud scoring with SHAP explanations",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/health', response_model=HealthResponse)
def health():
    uptime = time.time() - app_state['start_time']
    return HealthResponse(
        status="ok",
        model_version="xgboost_ensemble_v1",
        uptime_seconds=round(uptime, 2),
        last_prediction_timestamp=app_state['last_prediction_time']
    )

@app.get('/metrics', response_model=MetricsResponse)
def metrics():
    n = app_state['total_predictions']
    return MetricsResponse(
        total_predictions=n,
        fraud_flagged=app_state['fraud_flagged'],
        fraud_rate=round(app_state['fraud_flagged'] / n, 4) if n > 0 else 0.0,
        avg_xgb_probability=round(app_state['xgb_proba_sum'] / n, 4) if n > 0 else 0.0,
        avg_ensemble_score=round(app_state['ensemble_score_sum'] / n, 4) if n > 0 else 0.0,
    )

@app.post('/predict', response_model=PredictionResponse)
def predict(transaction: TransactionInput):
    try:
        raw = transaction.dict()
 
        # 1. Feature engineering -- builds card_id, hour_sin/cos, amt_log,
        #    amt_zscore_card. card_id/ProductCD/P_emaildomain/M-cols still
        #    raw strings at this point.
        df = engineer_features(
            raw,
            ml_models['feature_columns'],
            ml_models['card_stats']
        )
 
        # 2. Encoding (fit on train only -- never refit at inference time)
        df = apply_encoders(
            df,
            ml_models['target_encoder'],
            ml_models['label_encoders'],
            ml_models['te_cols']
        )
 
        # 3. Exact column order match
        df = align_columns(df, ml_models['feature_columns'])

        # ── TEMP DEBUG ──
        print("Columns match expected order:", df.columns.tolist() == ml_models['feature_columns'])
        print("Shape:", df.shape)
        print("First 10 values:", df.iloc[0][:10].to_dict())
        print("Any NaN values:", df.isna().sum().sum())
        print("Dtypes:", df.dtypes.value_counts())
        # ── END DEBUG ──

        # 4. XGBoost prediction
        xgb_proba = float(ml_models['xgb'].predict_proba(df)[:, 1][0])
 
        # 5. Autoencoder reconstruction error
        scaler = ml_models['scaler']
        cfg = ml_models['ensemble_config']
 
        df_scaled = scaler.transform(df)
        df_clipped = np.clip(df_scaled, -cfg['ae_clip_val'], cfg['ae_clip_val'])
        reconstructed = ml_models['autoencoder'].predict(df_clipped, verbose=0)
        ae_error = float(np.mean((df_clipped - reconstructed) ** 2))
 
        # 6. Normalize AE error using the SAME p1/p99 percentiles from training
        ae_error_clipped = np.clip(ae_error, cfg['ae_norm_min'], cfg['ae_norm_max'])
        ae_norm = (ae_error_clipped - cfg['ae_norm_min']) / (cfg['ae_norm_max'] - cfg['ae_norm_min'])
 
        # 7. Ensemble score
        ensemble_score = cfg['w1_xgb'] * xgb_proba + cfg['w2_ae'] * ae_norm
        fraud_flag = int(ensemble_score >= cfg['ensemble_threshold'])
 
        # 8. SHAP explanation -- top 5 contributing features
        shap_vals = ml_models['shap_explainer'].shap_values(df)
        feature_names = ml_models['feature_columns']
        top_idx = np.argsort(np.abs(shap_vals[0]))[::-1][:5]
        shap_top5 = {feature_names[i]: round(float(shap_vals[0][i]), 4) for i in top_idx}
 
        top_feature = feature_names[top_idx[0]]
        direction = "increases" if shap_vals[0][top_idx[0]] > 0 else "decreases"
        explanation = (
            f"Primary driver: {top_feature} {direction} fraud risk. "
            f"{'Flagged' if fraud_flag else 'Not flagged'} "
            f"with ensemble score {ensemble_score:.4f} "
            f"(threshold {cfg['ensemble_threshold']:.4f})."
        )
 
        # ── Update in-memory running metrics ──
        app_state['total_predictions'] += 1
        app_state['fraud_flagged'] += fraud_flag
        app_state['xgb_proba_sum'] += xgb_proba
        app_state['ensemble_score_sum'] += ensemble_score
        app_state['last_prediction_time'] = datetime.utcnow().isoformat()
 
        return PredictionResponse(
            fraud=fraud_flag,
            fraud_probability=round(xgb_proba, 4),
            xgb_probability=round(xgb_proba, 4),
            ae_reconstruction_error=round(ae_error, 6),
            ensemble_score=round(ensemble_score, 4),
            threshold=cfg['ensemble_threshold'],
            shap_top5=shap_top5,
            explanation=explanation
        )
 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))