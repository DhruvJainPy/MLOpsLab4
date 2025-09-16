import os
import yaml
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.metrics import accuracy_score, f1_score

with open("params.yaml", 'r') as f:
    params = yaml.safe_load(f)

processed_dir = 'data/processed'
test_data_path = os.path.join(processed_dir, 'test.csv')
target_col = params['target_column']
model_dir = 'models/'
reports_dir = 'experiments/'
metrics_path = os.path.join(reports_dir, 'results.json')
os.makedirs(reports_dir, exist_ok=True)

test_df = pd.read_csv(test_data_path)
X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

results = []

for model_file in os.listdir(model_dir):
    if not model_file.endswith(".pkl"):
        continue

    model_path = os.path.join(model_dir, model_file)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='macro')

    metrics = {
        "model_file": model_file,
        "accuracy": accuracy,
        "f1_score": f1
    }

    print(f"{model_file} → Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    results.append(metrics)

with open(metrics_path, 'w') as f:
    json.dump(results, f, indent = 4)
    
with open("metrics.json", "w") as f:
    json.dump(results, f, indent = 4)

print("All results saved to", metrics_path)
print("All results saved to metrics.json")
