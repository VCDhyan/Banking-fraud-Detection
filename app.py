#!/usr/bin/env python3
"""
Banking Fraud Detection Web Application
A Flask web app for interactive fraud detection with file upload and visualization.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import io
import base64

app = Flask(__name__)
app.secret_key = 'fraud_detection_secret_key_2024'

# Ensure reports directory exists
os.makedirs('reports', exist_ok=True)
os.makedirs('static/uploads', exist_ok=True)

def detect_fraud_anomalies(data, contamination=0.05):
    """Detect fraud using Isolation Forest."""
    # Prepare features
    features = ['amount']
    if 'amount' not in data.columns:
        raise ValueError("Data must contain 'amount' column")

    X = data[features].values

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )

    # Predict anomalies (-1 for anomaly, 1 for normal)
    predictions = iso_forest.fit_predict(X_scaled)

    # Convert to fraud labels (1 for fraud, 0 for normal)
    fraud_predictions = (predictions == -1).astype(int)

    return fraud_predictions, iso_forest, scaler

def generate_visualization(data, predictions, filename='fraud_visualization.png'):
    """Generate fraud detection visualization."""
    plt.figure(figsize=(12, 8))

    # Scatter plot
    plt.subplot(2, 2, 1)
    colors = ['red' if pred == 1 else 'blue' for pred in predictions]
    plt.scatter(data['amount'], range(len(data)), c=colors, alpha=0.6, s=20)
    plt.xlabel('Transaction Amount ($)')
    plt.ylabel('Transaction Index')
    plt.title('Transaction Amounts\n(Red = Fraud, Blue = Normal)')
    plt.grid(True, alpha=0.3)

    # Amount distribution
    plt.subplot(2, 2, 2)
    plt.hist(data['amount'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Transaction Amount ($)')
    plt.ylabel('Frequency')
    plt.title('Transaction Amount Distribution')
    plt.grid(True, alpha=0.3)

    # Fraud vs Normal amounts
    plt.subplot(2, 2, 3)
    fraud_amounts = data[predictions == 1]['amount']
    normal_amounts = data[predictions == 0]['amount']

    plt.hist([normal_amounts, fraud_amounts], bins=30, alpha=0.7,
             label=['Normal', 'Fraud'], color=['blue', 'red'])
    plt.xlabel('Transaction Amount ($)')
    plt.ylabel('Frequency')
    plt.title('Fraud vs Normal Transaction Amounts')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Box plot
    plt.subplot(2, 2, 4)
    box_data = [normal_amounts, fraud_amounts]
    plt.boxplot(box_data, labels=['Normal', 'Fraud'])
    plt.ylabel('Transaction Amount ($)')
    plt.title('Transaction Amount Box Plot')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'reports/{filename}', dpi=100, bbox_inches='tight')
    plt.close()

    return f'reports/{filename}'

def generate_confusion_matrix_plot(y_true, y_pred, filename='confusion_matrix.png'):
    """Generate confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'reports/{filename}', dpi=100, bbox_inches='tight')
    plt.close()

    return f'reports/{filename}'

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload and fraud detection."""
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file uploaded')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)

        if file and file.filename.endswith('.csv'):
            try:
                # Read CSV file
                data = pd.read_csv(file)

                # Check if required columns exist
                if 'amount' not in data.columns:
                    flash('CSV must contain an "amount" column')
                    return redirect(request.url)

                # Add fraud labels if they exist, otherwise assume all are normal for testing
                if 'is_fraud' not in data.columns:
                    data['is_fraud'] = 0  # Assume no fraud for uploaded data

                # Run fraud detection
                predictions, model, scaler = detect_fraud_anomalies(data)

                # Calculate metrics
                if 'is_fraud' in data.columns:
                    accuracy = accuracy_score(data['is_fraud'], predictions)
                    report = classification_report(data['is_fraud'], predictions, output_dict=True)
                else:
                    accuracy = None
                    report = None

                # Generate visualizations
                viz_file = generate_visualization(data, predictions)
                cm_file = generate_confusion_matrix_plot(data['is_fraud'], predictions) if 'is_fraud' in data.columns else None

                # Prepare results
                results = {
                    'total_transactions': len(data),
                    'detected_fraud': int(predictions.sum()),
                    'accuracy': accuracy,
                    'report': report,
                    'visualization': viz_file,
                    'confusion_matrix': cm_file
                }

                return render_template('results.html', results=results, data=data.head(10).to_dict('records'))

            except Exception as e:
                flash(f'Error processing file: {str(e)}')
                return redirect(request.url)
        else:
            flash('Please upload a CSV file')
            return redirect(request.url)

    return render_template('upload.html')

@app.route('/demo')
def demo():
    """Run demo with synthetic data."""
    try:
        # Generate synthetic data
        np.random.seed(42)
        n_samples = 1000
        n_frauds = 50

        # Normal transactions
        normal_amounts = np.random.normal(300, 100, n_samples - n_frauds)
        normal_amounts = np.clip(normal_amounts, 10, 1000)

        # Fraudulent transactions (higher amounts)
        fraud_amounts = np.random.normal(800, 200, n_frauds)
        fraud_amounts = np.clip(fraud_amounts, 500, 2000)

        # Combine data
        all_amounts = np.concatenate([normal_amounts, fraud_amounts])
        is_fraud = np.concatenate([np.zeros(n_samples - n_frauds), np.ones(n_frauds)])

        data = pd.DataFrame({
            'amount': all_amounts,
            'is_fraud': is_fraud.astype(int)
        })

        # Run fraud detection
        predictions, model, scaler = detect_fraud_anomalies(data)

        # Calculate metrics
        accuracy = accuracy_score(data['is_fraud'], predictions)
        report = classification_report(data['is_fraud'], predictions, output_dict=True)

        # Generate visualizations
        viz_file = generate_visualization(data, predictions, 'demo_visualization.png')
        cm_file = generate_confusion_matrix_plot(data['is_fraud'], predictions, 'demo_confusion_matrix.png')

        # Prepare results
        results = {
            'total_transactions': len(data),
            'detected_fraud': int(predictions.sum()),
            'accuracy': accuracy,
            'report': report,
            'visualization': viz_file,
            'confusion_matrix': cm_file
        }

        return render_template('results.html', results=results, data=data.head(10).to_dict('records'))

    except Exception as e:
        flash(f'Error running demo: {str(e)}')
        return redirect(url_for('index'))

@app.route('/reports/<filename>')
def get_report(filename):
    """Serve report images."""
    return send_file(f'reports/{filename}', mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)