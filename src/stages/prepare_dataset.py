from sklearn.preprocessing import StandardScaler, OrdinalEncoder, PowerTransformer
import pandas as pd
import yaml
import sys
import os
sys.path.append(os.getcwd())

from src.loggers import get_logger

def load_config(config_path):
    with open(config_path) as conf_file:
        return yaml.safe_load(conf_file)

def clear_data(path2data):
    df = pd.read_csv(path2data)
    cat_columns = ['Make', 'Model', 'Style', 'Fuel_type', 'Transmission']
    
    # Очистка аномалий
    df = df.drop(df[(df.Year < 2021) & (df.Distance < 1100)].index)
    df = df.drop(df[(df.Distance > 1e6)].index)
    df = df.drop(df[df["Engine_capacity(cm3)"] < 200].index)
    df = df.drop(df[df["Engine_capacity(cm3)"] > 5000].index)
    df = df.drop(df[(df["Price(euro)"] < 101)].index)
    df = df.drop(df[df["Price(euro)"] > 1e5].index)
    df = df.drop(df[df.Year < 1971].index)
    
    df = df.reset_index(drop=True)  
    ordinal = OrdinalEncoder()
    df_ordinal = pd.DataFrame(ordinal.fit_transform(df[cat_columns]), columns=cat_columns)
    df[cat_columns] = df_ordinal[cat_columns]
    return df

def featurize(dframe, config) -> None:
    logger = get_logger('FEATURIZE')
    logger.info('Create features')
    dframe['Distance_by_year'] = dframe['Distance'] / (2022 - dframe['Year'])
    dframe['age'] = 2024 - dframe['Year']
    mean_engine_cap = dframe.groupby('Style')['Engine_capacity(cm3)'].mean()
    dframe['eng_cap_diff'] = dframe.apply(lambda row: abs(row['Engine_capacity(cm3)'] - mean_engine_cap[row['Style']]), axis=1)

    features_path = config['featurize']['features_path']
    dframe.to_csv(features_path, index=False)

if __name__ == "__main__":
    config = load_config("./src/config.yaml")
    df_prep = clear_data(config['data_load']['dataset_csv'])
    featurize(df_prep, config)