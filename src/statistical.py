import pandas as pd
import numpy as np

def zscore_flag(df, threshold=3.0):
    """Flag rows where ANY feature has Z-score > threshold."""
    features = [c for c in df.columns if c not in ['Class', 'Amount_scaled', 'Time_scaled']]
    z_scores = np.abs((df[features] - df[features].mean()) / df[features].std())
    df['zscore_anomaly'] = (z_scores > threshold).any(axis=1).astype(int)
    df['max_zscore'] = z_scores.max(axis=1)
    return df

def iqr_flag(df):
    """Flag rows where Amount is outside IQR bounds."""
    Q1 = df['Amount_scaled'].quantile(0.25)
    Q3 = df['Amount_scaled'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df['iqr_anomaly'] = ((df['Amount_scaled'] < lower) | 
                         (df['Amount_scaled'] > upper)).astype(int)
    return df

def combined_risk_score(df):
    """Combine flags into a single risk score 0–100."""
    df['risk_score'] = (df['zscore_anomaly'] * 50 + 
                        df['iqr_anomaly'] * 30 + 
                        df['max_zscore'].clip(0, 10) * 2).clip(0, 100)
    df['risk_label'] = pd.cut(df['risk_score'], 
                               bins=[-1, 30, 60, 100], 
                               labels=['Low', 'Medium', 'High'])
    return df