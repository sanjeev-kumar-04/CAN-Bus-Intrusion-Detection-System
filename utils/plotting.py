import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_training_history(train_losses, val_losses, metrics_history, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_dir}/loss_curve.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(metrics_history['accuracy'], label='Accuracy', marker='o')
    plt.title('Model Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_dir}/accuracy.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    for metric in ['precision', 'recall', 'f1']:
        plt.plot(metrics_history[metric], label=metric.capitalize(), marker='o')
    plt.title('Training Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_dir}/metrics.png')
    plt.close()


def plot_confusion_matrix(cm, save_dir='plots'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{save_dir}/confusion_matrix.png')
    plt.close()