import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    
    # Scale Amount and Time (V1-V28 are already PCA-scaled)
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_scaled'] = scaler.fit_transform(df[['Time']])
    
    # Drop original Amount and Time
    df.drop(columns=['Amount', 'Time'], inplace=True)
    
    return df

if __name__ == '__main__':
    df = load_and_preprocess('../data/creditcard.csv')
    print(df.head())
    print(f"Shape: {df.shape}")