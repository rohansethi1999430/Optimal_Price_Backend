from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

# Load the trained model
model_path = '/Users/vidushichauhan/Desktop/product_price_model_ridge.pkl'
model = joblib.load(model_path)

# Function to preprocess input data
def preprocess_data(input_data):
    # Convert categorical data to numerical using frequency encoding
    freq_encoding = {
        'Industrial & Scientific': 4,  # Example frequency encoding; adjust based on actual data
        'Other Category': 1
    }
    input_data['categoryName'] = freq_encoding.get(input_data['categoryName'], 0)
    # Convert the data into a DataFrame
    df = pd.DataFrame([input_data])
    # Ensure all columns are in the correct order and format
    df = df[['price', 'categoryName', 'stars', 'reviews']]
    return df

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        input_data = request.json
        # Preprocess the input data
        processed_data = preprocess_data(input_data)
        # Make a prediction using the loaded model
        prediction = model.predict(processed_data)
        # Return the prediction as a JSON response
        return jsonify({'bestSellerPrediction': bool(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
