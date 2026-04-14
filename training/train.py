import torch
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import OneCycleLR

from models.loss import FocalLoss
from utils.metrics import format_metrics
from utils.plotting import plot_training_history

def train_model(model, train_loader, val_loader, class_weights, epochs=15, patience=5):
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.002,
        epochs=epochs,
        steps_per_epoch=len(train_loader)
    )

    device = next(model.parameters()).device

    train_losses = []
    val_losses = []

    metrics_history = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch).squeeze()

            loss = criterion(output, y_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0
        val_preds = []
        val_true = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                output = model(X_batch).squeeze()
                loss = criterion(output, y_batch)

                val_loss += loss.item()
                val_preds.extend((output > 0.5).cpu().numpy())
                val_true.extend(y_batch.cpu().numpy())

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        metrics = classification_report(val_true, val_preds, zero_division=1, output_dict=True)

        metrics_history['accuracy'].append(metrics['accuracy'])
        metrics_history['precision'].append(metrics['macro avg']['precision'])
        metrics_history['recall'].append(metrics['macro avg']['recall'])
        metrics_history['f1'].append(metrics['macro avg']['f1-score'])

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(format_metrics(metrics))

        plot_training_history(train_losses, val_losses, metrics_history)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, 'best_model.pth')

        else:
            patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    return train_losses, val_losses, metrics_history