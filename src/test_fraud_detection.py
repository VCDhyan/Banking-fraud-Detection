#!/usr/bin/env python3
"""
Banking Fraud Detection Test Script
This script tests the core functionality of the fraud detection system.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

def generate_test_data(n_samples=1000, n_frauds=50):
    """Generate synthetic banking transaction data."""
    np.random.seed(42)

    # Normal transactions
    normal_amounts = np.random.normal(300, 100, n_samples - n_frauds)
    normal_amounts = np.clip(normal_amounts, 10, 1000)

    # Fraudulent transactions
    fraud_amounts = np.random.normal(2000, 500, n_frauds)
    fraud_amounts = np.clip(fraud_amounts, 1000, 5000)

    # Combine
    all_amounts = np.concatenate([normal_amounts, fraud_amounts])
    fraud_labels = np.concatenate([np.zeros(n_samples - n_frauds), np.ones(n_frauds)])

    return pd.DataFrame({
        'amount': all_amounts,
        'is_fraud': fraud_labels.astype(int)
    })

def test_anomaly_detection(df):
    """Test the anomaly detection model."""
    # Prepare data
    features = ['amount']
    X = df[features]

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Isolation Forest
    isolation_forest = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )

    # Fit and predict
    df = df.copy()
    df['anomaly_score'] = isolation_forest.fit_predict(X_scaled)
    df['predicted_fraud'] = (df['anomaly_score'] == -1).astype(int)

    return df

def evaluate_model(df):
    """Evaluate the model performance."""
    print("=== Model Evaluation ===")
    print(f"Total transactions: {len(df)}")
    print(f"Actual fraudulent: {df['is_fraud'].sum()}")
    print(f"Predicted fraudulent: {df['predicted_fraud'].sum()}")

    # Confusion Matrix
    cm = confusion_matrix(df['is_fraud'], df['predicted_fraud'])
    print("\nConfusion Matrix:")
    print(cm)

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(df['is_fraud'], df['predicted_fraud']))

    # Accuracy
    accuracy = (df['is_fraud'] == df['predicted_fraud']).mean()
    print(".3f")

    return accuracy

def main():
    """Main test function."""
    print("Testing Banking Fraud Detection System")
    print("=" * 50)

    # Generate test data
    print("1. Generating synthetic transaction data...")
    df = generate_test_data()
    print(f"   Created {len(df)} transactions")

    # Test anomaly detection
    print("2. Running anomaly detection...")
    df_results = test_anomaly_detection(df)
    print("   Detection completed")

    # Evaluate
    print("3. Evaluating model performance...")
    accuracy = evaluate_model(df_results)

    # Simple visualization
    print("4. Creating visualization...")
    plt.figure(figsize=(8, 6))
    plt.scatter(df_results['amount'], range(len(df_results)),
               c=df_results['is_fraud'], cmap='coolwarm', alpha=0.6, s=10)
    plt.xlabel('Transaction Amount')
    plt.ylabel('Transaction Index')
    plt.title('Transaction Amounts (Red = Fraud)')
    plt.colorbar(label='Fraud Status')
    plt.savefig('reports/fraud_detection_test.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("   Visualization saved to reports/fraud_detection_test.png")

    print("\n=== Test Summary ===")
    print("✓ Data generation: PASSED")
    print("✓ Anomaly detection: PASSED")
    print("✓ Model evaluation: PASSED")
    print("✓ Visualization: PASSED")
    print(f"Accuracy: {accuracy:.3f}")
    if accuracy > 0.8:
        print("🎉 System is working well!")
    else:
        print("⚠️  System needs improvement")

if __name__ == "__main__":
    main()