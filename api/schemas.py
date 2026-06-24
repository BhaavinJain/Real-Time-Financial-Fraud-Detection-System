from pydantic import BaseModel , Field , create_model
from typing import Dict , Optional

ALL_FEATURE_COLUMNS = ['TransactionAmt', 'ProductCD', 'addr1', 'addr2', 'P_emaildomain', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D10', 'D11', 'D15', 'M1', 'M2', 'M3', 'M4', 'M6', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V65', 'V66', 'V67', 'V68', 'V69', 'V70', 'V71', 'V72', 'V73', 'V74', 'V75', 'V76', 'V77', 'V78', 'V79', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V86', 'V87', 'V88', 'V89', 'V90', 'V91', 'V92', 'V93', 'V94', 'V95', 'V96', 'V97', 'V98', 'V99', 'V100', 'V101', 'V102', 'V103', 'V104', 'V105', 'V106', 'V107', 'V108', 'V109', 'V110', 'V111', 'V112', 'V113', 'V114', 'V115', 'V116', 'V117', 'V118', 'V119', 'V120', 'V121', 'V122', 'V123', 'V124', 'V125', 'V126', 'V127', 'V128', 'V129', 'V130', 'V131', 'V132', 'V133', 'V134', 'V135', 'V136', 'V137', 'V279', 'V280', 'V281', 'V282', 'V283', 'V284', 'V285', 'V286', 'V287', 'V288', 'V289', 'V290', 'V291', 'V292', 'V293', 'V294', 'V295', 'V296', 'V297', 'V298', 'V299', 'V300', 'V301', 'V302', 'V303', 'V304', 'V305', 'V306', 'V307', 'V308', 'V309', 'V310', 'V311', 'V312', 'V313', 'V314', 'V315', 'V316', 'V317', 'V318', 'V319', 'V320', 'V321', 'hour_sin', 'hour_cos', 'card_id', 'amt_log', 'amt_zscore_card']

COMPUTED_COLUMNS = {'card_id', 'hour_sin', 'hour_cos', 'amt_log', 'amt_zscore_card'}

REQUIRED_FIELDS = {
    'TransactionAmt' : (float , Field(... , gt = 0)),
    'TransactionDT' : (int , Field(... , description = "Seconds since reference timestamp, will be converted into sin/ cos hour for model")),
    'ProductCD':      (str, ...),
    'card1':          (int, ...),
    'card4':          (str, ...),
    'card6':          (str, ...),
}

# Fields that can be missing in real transaction data 
# card2/3/5 come from card verification systems that don't always respond
# M columns are vesta engineered data

DEFAULTABLE_FIELDS = {
    'card2':         (float, -999.0),
    'card3':         (float, -999.0),
    'card5':         (float, -999.0),
    'addr1':         (float, -999.0),
    'addr2':         (float, -999.0),
    'P_emaildomain': (str, "unknown"),
    'M1': (str, "missing"),
    'M2': (str, "missing"),
    'M3': (str, "missing"),
    'M4': (str, "missing"),
    'M6': (str, "missing"),
}

EXPLICIT_FIELDS = {**REQUIRED_FIELDS, **DEFAULTABLE_FIELDS}

# All the other remaining V columns (also vesta engineered)

auto_fields = {
    col: (float, -999.0)
    for col in ALL_FEATURE_COLUMNS
    if col not in EXPLICIT_FIELDS and col not in COMPUTED_COLUMNS
}

all_fields = {**EXPLICIT_FIELDS, **auto_fields}

# Pydantic model for taking initializing input fields

TransactionInput = create_model(
    'TransactionInput',
    **all_fields
)

# Prediction Pydantic model

class PredictionResponse(BaseModel):
    fraud: int
    fraud_probability: float
    xgb_probability: float
    ae_reconstruction_error: float
    ensemble_score: float
    threshold: float
    shap_top5: Dict[str, float]
    explanation: str
 
 
class HealthResponse(BaseModel):
    status: str
    model_version: str
    uptime_seconds: float
    last_prediction_timestamp: Optional[str]
 
 
class MetricsResponse(BaseModel):
    total_predictions: int
    fraud_flagged: int
    fraud_rate: float
    avg_xgb_probability: float
    avg_ensemble_score: float