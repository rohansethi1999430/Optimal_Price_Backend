from flask import Flask, request, jsonify
import joblib as job
import pandas as pd
from flask_cors import CORS

# Load the model and feature names
model_path = 'product_price_model_ridge.pkl'
feature_names_path = 'product_price_model_ridge_features.pkl'
model = job.load(model_path)
feature_names = job.load(feature_names_path)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Convert input data to DataFrame
    input_data = pd.DataFrame([data])
   
    # Preprocess the input data
    input_data = pd.get_dummies(input_data, columns=['categoryName', 'title'], drop_first=True)
   
    # Find missing columns
    missing_cols = set(feature_names) - set(input_data.columns)
    
    # Add missing columns with value 0
    for col in missing_cols:
        input_data[col] = 0
    
    # Reorder columns to match the training data
    input_data = input_data[feature_names]
   
    # Predict the price
    prediction = model.predict(input_data)
   
    # Return the prediction
    return jsonify({'predicted_price': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
