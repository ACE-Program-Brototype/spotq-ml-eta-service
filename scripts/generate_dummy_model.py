# scripts/generate_dummy_model.py
import os
import xgboost as xgb
import numpy as np

os.makedirs("app/infrastructure/ml_models/artifacts", exist_ok=True)

X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y = np.array([100.0, 200.0])
dtrain = xgb.DMatrix(X, label=y)

booster = xgb.train({"objective": "reg:squarederror"}, dtrain, num_boost_round=1)
booster.save_model("app/infrastructure/ml_models/artifacts/eta_model_v1.json")
print("Successfully generated valid XGBoost model artifact.")