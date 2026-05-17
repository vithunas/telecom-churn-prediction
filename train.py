import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb
import joblib

def main():
    print("Initializing SparkSession...")
    # 1. Use PySpark to load and display the CSV
    spark = SparkSession.builder \
        .appName("TelcoChurn") \
        .master("local[*]") \
        .getOrCreate()
    
    print("Loading data via PySpark...")
    spark_df = spark.read.csv("Telco-Customer-Churn.csv", header=True, inferSchema=True)
    print("Data sample from PySpark:")
    spark_df.show(5)
    
    print("Converting to Pandas DataFrame...")
    df = spark_df.toPandas()
    
    print("Closing SparkSession...")
    spark.stop()
    
    # 2. Feature Engineering in Pandas
    print("Preprocessing data...")
    # Convert TotalCharges to float, coerce errors to NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    
    X = df.drop(['customerID', 'Churn'], axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    categorical_cols = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", 
                        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", 
                        "TechSupport", "StreamingTV", "StreamingMovies", "Contract", 
                        "PaperlessBilling", "PaymentMethod"]
    numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    
    # Create scikit-learn ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Train XGBoost with GridSearchCV
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', xgb_clf)])
    
    param_grid = {
        'classifier__max_depth': [3, 4, 5],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__n_estimators': [100, 200, 300]
    }
    
    print("Running GridSearchCV for XGBoost...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    accuracy = best_model.score(X_test, y_test)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # 4. Save the model and pipeline
    print("Saving model and pipeline to model.joblib...")
    joblib.dump(best_model, "model.joblib")
    
    print("Training complete.")

if __name__ == "__main__":
    main()
