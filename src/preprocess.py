import os
import yaml
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

with open("params.yaml", 'r') as f:
    params = yaml.safe_load(f)

raw_data_path = params['dataset']
processed_dir = 'data/processed'
train_path = os.path.join(processed_dir, 'train.csv')
test_path = os.path.join(processed_dir, 'test.csv')
target_col = params['target_column']

os.makedirs(processed_dir, exist_ok = True)

print(f"Loading raw data from {raw_data_path}")
df = pd.read_csv(raw_data_path)

X = df.drop(columns = [target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size = params['training']['test_size'], 
    random_state = params['training']['random_state']
)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

scaling = params['preprocessing']['scale']
if scaling:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


X_train_processed = pd.DataFrame(X_train_scaled, index = X_train.index, columns = X_train.columns)
X_test_processed = pd.DataFrame(X_test_scaled, index = X_test.index, columns = X_test.columns)


train_df = pd.concat([
    X_train_processed, 
    pd.Series(y_train_encoded, name = target_col, index = X_train.index)
], axis = 1)

test_df = pd.concat([
    X_test_processed, 
    pd.Series(y_test_encoded, name = target_col, index = X_test.index)
], axis = 1)

train_df.to_csv(train_path, index = False)
test_df.to_csv(test_path, index = False)