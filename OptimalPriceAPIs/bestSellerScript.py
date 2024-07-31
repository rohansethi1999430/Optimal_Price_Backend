import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load the data
df = pd.read_csv('/content/amz_ca_total_products_data_processed.csv')  # Replace with the correct path to your CSV file

# Sample the data
df_sampled = df.sample(frac=0.1, random_state=42)

# Select relevant columns
df_sampled = df_sampled[['stars', 'reviews', 'price', 'categoryName', 'isBestSeller', 'boughtInLastMonth']]

# Drop rows with missing values
df_sampled.dropna(inplace=True)

# Convert 'isBestSeller' to boolean
df_sampled['isBestSeller'] = df_sampled['isBestSeller'].astype(bool)

# One-hot encode categorical features
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(df_sampled[['categoryName', 'isBestSeller']])
encoded_features_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['categoryName', 'isBestSeller']))

# Combine the encoded features with the numerical features
numerical_features = df_sampled[['stars', 'reviews', 'boughtInLastMonth']]
preprocessed_df = pd.concat([numerical_features.reset_index(drop=True), encoded_features_df.reset_index(drop=True)], axis=1)

# Define the target variable
target = df_sampled['price'].reset_index(drop=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(preprocessed_df, target, test_size=0.2, random_state=42)

# Train a RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Save the model and the encoder
joblib.dump(model, '/content/price_prediction_model.joblib')
joblib.dump(encoder, '/content/encoder.joblib')

# Function to predict optimal price
def predict_optimal_price(category_name, stars, reviews, current_price, is_best_seller, bought_in_last_month):
    # Load the model and encoder
    model = joblib.load('price_prediction_model.joblib')
    encoder = joblib.load('encoder.joblib')
    
    # Preprocess the input
    input_data = {
        'stars': [stars],
        'reviews': [reviews],
        'price': [current_price],
        'boughtInLastMonth': [bought_in_last_month],
        'categoryName': [category_name],
        'isBestSeller': [is_best_seller]
    }
    
    input_df = pd.DataFrame(input_data)
    encoded_input = encoder.transform(input_df[['categoryName', 'isBestSeller']])
    encoded_input_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(['categoryName', 'isBestSeller']))
    
    preprocessed_input = pd.concat([input_df[['stars', 'reviews', 'boughtInLastMonth']].reset_index(drop=True), encoded_input_df.reset_index(drop=True)], axis=1)
    
    # Predict the optimal price
    optimal_price = model.predict(preprocessed_input)
    return optimal_price[0]

# Example usage
example_input = {
    "category_name": "Industrial Scientific",
    "stars": 5,
    "reviews": 1000,
    "current_price": 49.99,
    "is_best_seller": False,
    "bought_in_last_month": 100
}

optimal_price = predict_optimal_price(**example_input)
print(f'Optimal Price: {optimal_price}')
