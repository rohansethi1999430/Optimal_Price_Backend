from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from scipy.optimize import minimize_scalar

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the trained regression model and encoder
regressor = joblib.load('/Users/vidushichauhan/Desktop/OptimalPrice/OptimalPriceAPIs/random_forest_regressor_model.pkl')
encoder = joblib.load('/Users/vidushichauhan/Desktop/OptimalPrice/OptimalPriceAPIs/encoder.pkl')

# Assume X.columns is available from the training code
# Load a dummy model and encoder to get the column names
dummy_df = pd.DataFrame(columns=['reviews', 'price', 'stars'])
X = pd.get_dummies(dummy_df)

def preprocess_input(features):
    """
    Preprocess the input features in the same way as the training data.
    
    :param features: A dictionary containing the input features.
    :return: Preprocessed features ready for prediction.
    """
    # Create a DataFrame from the input features
    input_df = pd.DataFrame([features])
    
    # Convert 'isBestSeller' to boolean if not already
    if isinstance(input_df['isBestSeller'].iloc[0], str):
        input_df['isBestSeller'] = input_df['isBestSeller'].apply(lambda x: x.lower() == 'true')
    
    # One-hot encode categorical features
    encoded_features = encoder.transform(input_df[['categoryName', 'isBestSeller']])
    encoded_features_df = pd.DataFrame(encoded_features.toarray(), columns=encoder.get_feature_names_out())
    
    # Combine the encoded features with the numerical features
    preprocessed_input = pd.concat([encoded_features_df, input_df[['reviews', 'price']].reset_index(drop=True)], axis=1)
    
    # Ensure columns match with the model input
    missing_cols = set(X.columns) - set(preprocessed_input.columns)
    for col in missing_cols:
        preprocessed_input[col] = 0
    preprocessed_input = preprocessed_input[X.columns]
    
    return preprocessed_input

def objective_function(price, input_features):
    """
    Objective function to minimize (negative rating) for optimization.
    
    :param price: The price to evaluate.
    :param input_features: A dictionary containing the input features except for the price.
    :return: The negative predicted rating.
    """
    input_features['price'] = price
    preprocessed_input = preprocess_input(input_features)
    rating = regressor.predict(preprocessed_input)
    return -rating[0]

@app.route('/find_optimal_price', methods=['POST'])
def find_optimal_price():
    """
    Find the optimal price that maximizes the predicted rating.
    
    :return: JSON response with the optimal price.
    """
    data = request.json
    initial_price = data.get('initial_price', 50.0)
    input_features = data['input_features']
    
    result = minimize_scalar(objective_function, bounds=(0.01, 1000), args=(input_features,), method='bounded')
    optimal_price = result.x

    return jsonify({'optimal_price': optimal_price})

if __name__ == '__main__':
    app.run(debug=True)
