import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
import pickle
import mlflow
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from src.stages.prepare_dataset import load_config

def train(config):
    df_train = pd.read_csv(config['data_split']['trainset_path'])
    X_train = df_train.drop(columns=['Price(euro)']).values
    y_train = df_train['Price(euro)'].values

    power_trans = PowerTransformer()
    y_train = power_trans.fit_transform(y_train.reshape(-1, 1))
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("linear model cars")
    with mlflow.start_run():
        if config['train']['model_type'] == "tree":
            pipe = Pipeline([('scaler', StandardScaler()), ('model', ExtraTreesRegressor(random_state=42))])
            params = {'model__n_estimators': config['train']['n_estimators']}
        else:
            pipe = Pipeline([('scaler', StandardScaler()), ('model', SGDRegressor(random_state=42))])
            params = {'model__alpha': config['train']['alpha']}
        
        clf = GridSearchCV(pipe, params, cv=config['train']['cv'], n_jobs=4)
        clf.fit(X_train, y_train.reshape(-1))
        best = clf.best_estimator_

        # Сохранение моделей
        os.makedirs("models", exist_ok=True)
        with open(config['train']['model_path'], "wb") as file:
            pickle.dump(best, file)
        with open(config['train']['power_path'], "wb") as file:
            pickle.dump(power_trans, file)

if __name__ == "__main__":
    config = load_config("./src/config.yaml")
    train(config)