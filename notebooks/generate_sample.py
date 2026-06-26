"""
Generates dashboard/sample_transactions.csv -- a small, realistic batch
of RAW transactions (pre-feature-engineering) for the Streamlit dashboard's
"Batch Review" tab demo button.

Pulls a balanced mix from the original IEEE-CIS raw CSVs so the sample
file contains genuine fraud and legitimate examples, with all columns
the API's TransactionInput schema expects (everything else the API
doesn't recognize gets dropped automatically -- predict() ignores
unexpected dict keys).

Run from inside notebooks/ (same relative paths as your other scripts):
    python generate_sample_transactions.py
"""

import pandas as pd
import numpy as np

train_txn = pd.read_csv(r'D:\Fraud-Detection\data\raw\train_transaction.csv')

# Columns the API's TransactionInput schema actually expects --
# matches the C1-C14, D1/D2/D3/D4/D10/D11/D15, M1-M4/M6, V1-V137,
# V279-V321 set that survived Day 2's >50%-missing column drop.
KEEP_COLS = (
    ['TransactionDT', 'TransactionAmt', 'ProductCD',
     'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
     'addr1', 'addr2', 'P_emaildomain']
    + [f'C{i}' for i in range(1, 15)]
    + ['D1', 'D2', 'D3', 'D4', 'D10', 'D11', 'D15']
    + ['M1', 'M2', 'M3', 'M4', 'M6']
    + [f'V{i}' for i in range(1, 138)]
    + [f'V{i}' for i in range(279, 322)]
)

KEEP_COLS = [c for c in KEEP_COLS if c in train_txn.columns]

# Pull a balanced, small sample: 10 fraud + 15 legit, shuffled
fraud_sample = train_txn[train_txn['isFraud'] == 1].sample(n=10, random_state=42)
legit_sample = train_txn[train_txn['isFraud'] == 0].sample(n=15, random_state=42)

sample = pd.concat([fraud_sample, legit_sample]).sample(frac=1, random_state=7).reset_index(drop=True)

# Keep isFraud as a hidden reference column (prefixed with _) so you can
# visually compare the dashboard's verdict against ground truth, without
# it being sent to the API (the dashboard code drops underscore-prefixed
# columns before building the payload -- see note below)
output = sample[KEEP_COLS].copy()
output['_true_label'] = sample['isFraud'].values

# Fill M-columns and P_emaildomain NaN with the same imputation strings
# used in training, so the sample is realistic and the API doesn't have
# to guess (it would default these anyway, this just makes it explicit)
for col in ['M1', 'M2', 'M3', 'M4', 'M6']:
    output[col] = output[col].fillna('missing')
output['P_emaildomain'] = output['P_emaildomain'].fillna('unknown')

output.to_csv('../dashboard/sample_transactions.csv', index=False)

print(f"Saved {len(output)} transactions to dashboard/sample_transactions.csv")
print(f"  Fraud: {(output['_true_label']==1).sum()}  |  Legit: {(output['_true_label']==0).sum()}")
print(f"  Columns: {output.shape[1]} (including _true_label reference column)")