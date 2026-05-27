import sys
import os
import json
import pickle
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

sys.path.append(os.getcwd())
from src.stages.prepare_dataset import load_config

def test(config):
    df_test = pd.read_csv(config['test']['testset_path'])
    X_test = df_test.drop(columns=['Price(euro)']).values
    y_test = df_test['Price(euro)'].values

    with open(config['test']['model_path'], "rb") as file:
        model = pickle.load(file)
    with open(config['test']['power_path'], "rb") as file:
        power_trans = pickle.load(file)

    y_pred = model.predict(X_test)
    y_pred_inverse = power_trans.inverse_transform(y_pred.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(y_test, y_pred_inverse))
    mae = mean_absolute_error(y_test, y_pred_inverse)
    r2 = r2_score(y_test, y_pred_inverse)

    # Сохранение метрик для DVC
    metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    os.makedirs("dvclive", exist_ok=True)
    with open("dvclive/metrics.json", "w") as f:
        json.dump(metrics, f)
        
    print(f"Metrics saved: {metrics}")

if __name__ == "__main__":
    config = load_config("./src/config.yaml")
    test(config)