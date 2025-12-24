import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

def train_model(n_estimators=100, random_state=42):
    """
    Train Random Forest model with MLflow tracking
    """
    
    print("="*60)
    print("Starting model training with MLflow tracking")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv("Telco-Customer-Churn-Processed.csv")
    print(f"Data loaded: {df.shape}")
    
    # Prepare data
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Enable autolog
    mlflow.sklearn.autolog()
    
    # Train model
    print(f"\nTraining with n_estimators={n_estimators}, random_state={random_state}")
    
    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        random_state=int(random_state),
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("Model training completed!")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    print("\nMetrics logged to MLflow!")
    print("Model saved to MLflow!")
    
    return accuracy

if __name__ == "__main__":
    # Get parameters from command line
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    random_state = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    
    # Train model
    accuracy = train_model(n_estimators, random_state)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print("="*60)
