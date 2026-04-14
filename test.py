import pandas as pd
import numpy as np
import torch
import time
import random
import joblib
from collections import deque

from models.model import CANIDS
from utils.preprocessing import hex_to_int, parse_timestamp

# ----------------------------
# SETTINGS
# ----------------------------
SEQUENCE_LENGTH = 5   # must match training
DELAY = 0.5           # seconds between messages
scaler = joblib.load("scaler.pkl")
# ----------------------------
# LOAD DATASETS
# ----------------------------
def load_datasets():
    columns = ['Timestamp', 'CAN_ID', 'DLC'] + [f'DATA{i}' for i in range(8)] + ['Flag']

    dfs = []

    files = [
        ('normal_run_data.txt', 'Normal'),
        ('DoS_dataset.csv', 'DoS'),
        ('Fuzzy_dataset.csv', 'Fuzzy'),
        ('Gear_dataset.csv', 'Gear'),
        ('RPM_dataset.csv', 'RPM')
    ]

    for file, attack_type in files:
        if file.endswith('.txt'):
            df = pd.read_csv(f"data/{file}", sep=r'\s+', names=columns, nrows=20000)
        else:
            df = pd.read_csv(f"data/{file}", names=columns, nrows=20000)

        df['Attack_Type'] = attack_type
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)

    return df


# ----------------------------
# PREPROCESS SINGLE ROW
# ----------------------------
def process_row(row):
    data = [hex_to_int(row[f'DATA{i}']) for i in range(8)]

    features = data + [
        parse_timestamp(row['Timestamp']),
        hex_to_int(row['DLC'])
    ]

    features = np.array(features, dtype=np.float32).reshape(1, -1)
    features = scaler.transform(features)
    features = np.clip(features, -3, 3)
    return features.flatten()


# ----------------------------
# LOAD MODEL
# ----------------------------
def load_model(device):
    model = CANIDS(input_dim=10).to(device)

    checkpoint = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    return model


# ----------------------------
# REALTIME SIMULATION
# ----------------------------
def run_simulation():
    print("\n🚀 Starting Real-Time CAN IDS Simulation...\n")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    df = load_datasets()
    model = load_model(device)

    buffer = deque(maxlen=SEQUENCE_LENGTH)

    for i in range(len(df)):
        row = df.iloc[i]

        features = process_row(row)
        buffer.append(features)

        print(f"\n📡 Incoming Message {i+1}")
        print(f"Type: {row['Attack_Type']} | Flag: {row['Flag']}")

        buffer.append(features)

        # If buffer not full → pad it (VERY IMPORTANT FIX)
        if len(buffer) < SEQUENCE_LENGTH:
            padded = list(buffer) + [buffer[0]] * (SEQUENCE_LENGTH - len(buffer))
        else:
            padded = list(buffer)

        sequence = np.array(padded)
        sequence = np.expand_dims(sequence, axis=0)

        X = torch.FloatTensor(sequence).to(device)

        with torch.no_grad():
            output = model(X).item()

        prediction = "🚨 ATTACK" if output > 0.5 else "✅ NORMAL"

        print(f"🔍 Confidence: {output:.4f}")
        print(f"🧠 Prediction: {prediction}")

        # Prepare input
        sequence = np.array(buffer)
        sequence = np.expand_dims(sequence, axis=0)

        X = torch.FloatTensor(sequence).to(device)

        # Prediction
        with torch.no_grad():
            output = model(X).item()

        prediction = "🚨 ATTACK" if output > 0.5 else "✅ NORMAL"
        if output > 0.5:
            print("\n🛑 ATTACK DETECTED — STOPPING SYSTEM\n")
            break

        print(f"🔍 Model Confidence: {output:.4f}")
        print(f"🧠 Prediction: {prediction}")

        # Delay to simulate real-time
        time.sleep(DELAY)


# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    run_simulation()