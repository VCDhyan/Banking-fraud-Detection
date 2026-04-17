#!/usr/bin/env python3
"""
Simple Banking Fraud Detection Test
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

def main():
    print("Testing Banking Fraud Detection")
    print("=" * 40)

    # Generate simple test data
    np.random.seed(42)
    normal = np.random.normal(300, 50, 100)
    fraud = np.random.normal(2000, 200, 10)
    data = np.concatenate([normal, fraud])
    labels = np.concatenate([np.zeros(100), np.ones(10)])

    df = pd.DataFrame({'amount': data, 'is_fraud': labels.astype(int)})

    # Test anomaly detection
    X = df[['amount']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(contamination=0.1, random_state=42)
    df['predicted'] = (model.fit_predict(X_scaled) == -1).astype(int)

    # Results
    accuracy = (df['is_fraud'] == df['predicted']).mean()
    print(f"Total samples: {len(df)}")
    print(f"Actual fraud: {df['is_fraud'].sum()}")
    print(f"Predicted fraud: {df['predicted'].sum()}")
    print(".3f")

    if accuracy > 0.8:
        print("✅ SUCCESS: Fraud detection is working!")
    else:
        print("⚠️  WARNING: Detection accuracy is low")

    print("\nClassification Report:")
    print(classification_report(df['is_fraud'], df['predicted']))

if __name__ == "__main__":
    main()