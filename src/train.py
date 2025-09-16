import os
import yaml
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from itertools import product

with open("params.yaml", 'r') as f:
    params = yaml.safe_load(f)

processed_dir = 'data/processed'
train_data_path = os.path.join(processed_dir, 'train.csv')
target_col = params['target_column']
model_dir = 'models/'
os.makedirs(model_dir, exist_ok=True)

train_df = pd.read_csv(train_data_path)
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]

model_type = params['model']['type']
param_grid = params['model']['params']

if model_type != 'RandomForest':
    raise ValueError(f"Unsupported model type: {model_type}")


for i, (n, d, r) in enumerate(product(param_grid['n_estimators'], 
                                      param_grid['max_depth'],
                                      [param_grid['random_state']])):
    model_params = {
        'n_estimators': n,
        'max_depth': d,
        'random_state': r
    }

    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)

    model_path = os.path.join(model_dir, f"rf_model_{i}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Saved model {i} with params {model_params} → {model_path}")
