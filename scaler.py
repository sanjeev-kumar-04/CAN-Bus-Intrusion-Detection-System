import pandas as pd
import joblib
from utils.preprocessing import process_data

columns = ['Timestamp', 'CAN_ID', 'DLC'] + [f'DATA{i}' for i in range(8)] + ['Flag']

dfs = []

for data_type in ['normal_run_data.txt'] + [f'{attack}_dataset.csv' for attack in ['DoS', 'Fuzzy', 'Gear', 'RPM']]:
    if data_type.endswith('.txt'):
        df = pd.read_csv(f"data/{data_type}", sep=r'\s+', names=columns, nrows=30000)
        df['Attack_Type'] = 'Normal'
    else:
        df = pd.read_csv(f"data/{data_type}", names=columns, nrows=30000)
        df['Attack_Type'] = data_type.split('_')[0]

    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

# 🔥 THIS GENERATES SCALER
_, _, scaler = process_data(df)

# 🔥 SAVE IT
joblib.dump(scaler, "scaler.pkl")

print("✅ scaler.pkl created successfully!")