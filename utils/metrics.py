def format_metrics(metrics):
    total_support = sum(metrics[label]['support'] for label in ['0.0', '1.0'])

    return (
        f"\n{'='*60}\n"
        f"{'Class':>10} {'Precision':>12} {'Recall':>10} {'F1-Score':>10} {'Support':>10}\n"
        f"{'-'*60}\n"
        f"{'Normal':>10} {metrics['0.0']['precision']:>12.4f} {metrics['0.0']['recall']:>10.4f} "
        f"{metrics['0.0']['f1-score']:>10.4f} {metrics['0.0']['support']:>10.0f}\n"
        f"{'Attack':>10} {metrics['1.0']['precision']:>12.4f} {metrics['1.0']['recall']:>10.4f} "
        f"{metrics['1.0']['f1-score']:>10.4f} {metrics['1.0']['support']:>10.0f}\n"
        f"{'-'*60}\n"
        f"{'Total':>10} {'':<12} {'':>10} {'':>10} {total_support:>10.0f}\n"
        f"{'Accuracy':>10} {'':<12} {'':>10} {metrics['accuracy']:>10.4f}\n"
        f"{'Macro Avg':>10} {metrics['macro avg']['precision']:>12.4f} "
        f"{metrics['macro avg']['recall']:>10.4f} {metrics['macro avg']['f1-score']:>10.4f}\n"
        f"{'Wtd Avg':>10} {metrics['weighted avg']['precision']:>12.4f} "
        f"{metrics['weighted avg']['recall']:>10.4f} {metrics['weighted avg']['f1-score']:>10.4f}\n"
        f"{'='*60}\n"
    )