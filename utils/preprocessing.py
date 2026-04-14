import numpy as np
from sklearn.preprocessing import StandardScaler

def hex_to_int(x):
    try:
        if isinstance(x, str):
            if x in ['R', 'T']:
                return 0
            return int(x, 16) if 'x' not in x.lower() else int(x, 16)
        return int(x)
    except:
        return 0

def parse_timestamp(ts):
    try:
        if isinstance(ts, str) and any(c.isalpha() for c in ts):
            return float(int(ts, 16))
        return float(ts)
    except:
        return 0.0

def process_data(df, sequence_length=5):
    features = []

    for _, row in df.iterrows():
        data = [hex_to_int(row[f'DATA{i}']) for i in range(8)]
        features.append(data + [parse_timestamp(row['Timestamp']), hex_to_int(row['DLC'])])

    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    X = np.clip(X, -3, 3)

    y = (df['Flag'] == 'T').astype(int).values

    sequences = []
    labels = []

    for i in range(len(X) - sequence_length + 1):
        sequences.append(X[i:i + sequence_length])
        labels.append(1 if any(y[i:i + sequence_length]) else 0)

    return np.array(sequences), np.array(labels), scaler