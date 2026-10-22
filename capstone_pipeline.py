import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, silhouette_score, confusion_matrix

def main():
    print("--- 1. Data Acquisition ---")
    url = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
    df = pd.read_csv(url)
    
    print("\n--- 2. Data Cleaning ---")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    df.drop('customerID', axis=1, inplace=True)
    
    print("\n--- 3. Unsupervised Learning: Customer Segmentation ---")
    segment_data = df[['tenure', 'MonthlyCharges']]
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(segment_data)
    sil_score = silhouette_score(segment_data, df['Cluster'])
    print(f"K-Means Silhouette Score: {sil_score:.4f}")
    
    print("\n--- 4. Supervised Learning: Churn Prediction ---")
    # Prepare data
    df.drop('Cluster', axis=1, inplace=True) # drop cluster for churn pred
    X = pd.get_dummies(df.drop('Churn', axis=1), drop_first=True)
    y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    y_pred = rf_model.predict(X_test_scaled)
    
    print("\n--- 5. Evaluation ---")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    print("\nCapstone Project Pipeline Executed Successfully.")

if __name__ == "__main__":
    main()
