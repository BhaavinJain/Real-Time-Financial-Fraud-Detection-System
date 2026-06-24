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

    