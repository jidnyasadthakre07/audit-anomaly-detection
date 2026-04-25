import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve

def plot_confusion_matrix(y_true, y_pred, save_path='../outputs/confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legit', 'Fraud'],
                yticklabels=['Legit', 'Fraud'])
    plt.title('Confusion Matrix — Isolation Forest')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved: {save_path}")

def plot_risk_distribution(df, save_path='../outputs/risk_distribution.png'):
    plt.figure(figsize=(8, 4))
    df['anomaly_score'].hist(bins=100, color='steelblue', edgecolor='none')
    plt.axvline(df['anomaly_score'].quantile(0.968), 
                color='red', linestyle='--', label='3.2% threshold')
    plt.title('Anomaly Score Distribution')
    plt.xlabel('Anomaly Score (higher = more suspicious)')
    plt.legend()
    plt.savefig(save_path)

def export_flagged_transactions(df, save_path='../outputs/flagged_transactions.csv'):
    flagged = df[df['if_anomaly'] == 1].copy()
    flagged = flagged.sort_values('anomaly_score', ascending=False)
    flagged.to_csv(save_path, index=False)
    print(f"Exported {len(flagged)} flagged transactions to {save_path}")
    return flagged