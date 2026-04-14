import pandas as pd
import numpy as np
import torch
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from models.model import CANIDS
from utils.preprocessing import process_data
from utils.dataset import CANDataset
from utils.plotting import plot_confusion_matrix, plot_training_history
from utils.helpers import save_sample_data
from utils.metrics import format_metrics
from training.train import train_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Loading data...")

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
        logger.info(f"Loaded {len(df)} from {data_type}")

    df = pd.concat(dfs, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    save_sample_data(df)

    X, y, scaler = process_data(df)

    num_normal = np.sum(y == 0)
    num_attack = np.sum(y == 1)

    import joblib
    joblib.dump(scaler, "scaler.pkl")

    class_weights = torch.tensor([1.0, num_normal / num_attack], dtype=torch.float32)

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125)

    train_loader = DataLoader(CANDataset(X_train, y_train), batch_size=256, shuffle=True)
    val_loader = DataLoader(CANDataset(X_val, y_val), batch_size=256)
    test_loader = DataLoader(CANDataset(X_test, y_test), batch_size=256)

    device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu")

    model = CANIDS(input_dim=10).to(device)

    train_model(model, train_loader, val_loader, class_weights, epochs=15, patience=5)

    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    test_preds = []
    test_true = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            output = model(X_batch)

            test_preds.extend((output > 0.5).cpu().numpy().flatten())
            test_true.extend(y_batch.numpy())

    metrics = classification_report(test_true, test_preds, output_dict=True)
    print(format_metrics(metrics))

    cm = confusion_matrix(test_true, test_preds)
    print(cm)

    plot_confusion_matrix(cm)

if __name__ == "__main__":
    main()