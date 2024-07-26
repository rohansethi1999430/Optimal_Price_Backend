import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib as job
import os

try:
    # Load the data
    print("Loading data...")
    data = pd.read_csv('/Users/vidushichauhan/Desktop/OptimalPrice/product.csv')
    print("Data loaded successfully.")
    
    # Preprocess the data
    print("Preprocessing data...")
    data = pd.get_dummies(data, columns=['categoryName', 'title'], drop_first=True)
    print("Data preprocessed successfully.")
    
    # Define features and target variable
    X = data.drop('price', axis=1)
    y = data['price']
    
    # Split the data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split successfully.")
    
    # Train the model
    print("Training model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model trained successfully.")
    
    # Save the model
    print("Saving model...")
    model_path = '/Users/vidushichauhan/Desktop/OptimalPrice/product_price_model.pkl'
    job.dump(model, model_path)
    print(f"Model saved successfully at {model_path}")
    
except Exception as e:
    print(f"An error occurred: {e}")


