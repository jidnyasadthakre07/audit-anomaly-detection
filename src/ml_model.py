import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def train_isolation_forest(df, contamination=0.032):
    """
    contamination = expected fraction of anomalies.
    3.2% is the empirically tuned value for this dataset.
    """
    features = [c for c in df.columns 
                if c not in ['Class', 'risk_score', 'risk_label', 
                              'zscore_anomaly', 'iqr_anomaly', 'max_zscore']]
    
    X = df[features]
    
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1           # Use all CPU cores
    )
    
    model.fit(X)
    
    # Isolation Forest returns -1 for anomaly, 1 for normal
    df['if_prediction'] = model.predict(X)
    df['if_anomaly'] = (df['if_prediction'] == -1).astype(int)
    df['anomaly_score'] = -model.score_samples(X)  # Higher = more anomalous
    
    # Save model
    joblib.dump(model, '../outputs/isolation_forest.pkl')
    
    return df, model

def evaluate_model(df):
    """Evaluate against ground truth Class label."""
    y_true = df['Class']
    y_pred = df['if_anomaly']
    
    print("=== Isolation Forest Results ===")
    print(classification_report(y_true, y_pred, 
                                  target_names=['Legit', 'Fraud']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    flagged_pct = y_pred.sum() / len(y_pred) * 100
    print(f"\nFlagged as anomalous: {y_pred.sum()} ({flagged_pct:.1f}% of records)")