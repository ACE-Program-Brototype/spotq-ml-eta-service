# scripts/generate_dummy_model.py
import os
import xgboost as xgb
import numpy as np
from app.infrastructure.ml_models.features import FEATURE_NAMES

os.makedirs("app/infrastructure/ml_models/artifacts", exist_ok=True)

# Generate a canonical 6-feature baseline model artifact
X = np.array([
    [1.0, 0.0, 12.0, 0.0, 0.0, 20.0],
    [4.0, 5.0, 19.0, 4.0, 0.0, 25.0],
    [2.0, 10.0, 20.0, 5.0, 1.0, 30.0],
], dtype=np.float32)
y = np.array([15.0, 25.0, 35.0], dtype=np.float32)

dtrain = xgb.DMatrix(X, label=y, feature_names=list(FEATURE_NAMES))

booster = xgb.train(
    {"objective": "reg:squarederror", "eval_metric": "mae"},
    dtrain,
    num_boost_round=5,
)
booster.set_attr(
    feature_contract_version="v1",
    dataset_mode="synthetic",
    model_version="eta_model_v1",
)
booster.save_model("app/infrastructure/ml_models/artifacts/eta_model_v1.json")
print("Successfully generated valid canonical 6-feature XGBoost model artifact.")