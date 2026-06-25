import numpy as np
import pandas as pd

AMOUNT_CLIP = 2769.81

# Function to build the card_id column

def build_card_id(raw : dict) -> str:
    card1 = raw.get('card1' , -999)                      # Looking for the key card1 , in dict raw , it is exists use that value else -999
    card2 = raw.get('card1' , -999.0)
    card3 = raw.get('card1' , -999.0)
    card4 = raw.get('card1' , 'unknown')
    card5 = raw.get('card1' , -999.0)
    card6 = raw.get('card1' , 'unknown')

    return f"{card1}_{card2}_{card3}_{card4}_{card5}_{card6}"

# Function to engineer all the necessary featrures

def engineer_features(raw : dict , feature_columns : list , card_stats : dict) -> pd.DataFrame:
    row = {}
    card_id_str = build_card_id(raw)
    row["card_id"] = card_id_str

    amt = float(raw.get('TransactionAmt', 0.0))
    amt_clipped = min(amt, AMOUNT_CLIP)
    row['TransactionAmt'] = amt_clipped
    row['amt_log'] = np.log1p(amt_clipped)

    stats = card_stats.get(card_id_str)

    if stats and stats.get('std') not in (0, None) and not pd.isna(stats.get('std')):
        row['amt_zscore_card'] = (amt_clipped - stats['mean']) / stats['std']
    else:
        row['amt_zscore_card'] = 0.0

    dt = float(raw.get('TransactionDT', 0))
    hour = (dt / 3600) % 24
    row['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    row['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    row['ProductCD']     = raw.get('ProductCD', 'unknown')
    row['P_emaildomain'] = raw.get('P_emaildomain', 'unknown')
    row['M1'] = raw.get('M1', 'missing')
    row['M2'] = raw.get('M2', 'missing')
    row['M3'] = raw.get('M3', 'missing')
    row['M4'] = raw.get('M4', 'missing')
    row['M6'] = raw.get('M6', 'missing')

    already_set = set(row.keys())
    for col in feature_columns:
        if col not in already_set:
            row[col] = raw.get(col, -999.0)
 
    return pd.DataFrame([row])
 
def apply_encoders(df : pd.DataFrame , target_encoder , label_encoders , te_cols : list) -> pd.DataFrame : 
    df = df.copy()
    df[te_cols] = target_encoder.transform(df[te_cols])

    for col, le in label_encoders.items():
        val = str(df[col].iloc[0])
        if val in le.classes_:
            df[col] = le.transform([val])[0]
        else:
            df[col] = -1  # unseen category sentinel
 
    return df

def align_columns(df: pd.DataFrame, expected_columns: list) -> pd.DataFrame:
    for col in expected_columns:
        if col not in df.columns:
            df[col] = -999.0
    return df[expected_columns]