import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# Load and preprocess data
df = pd.read_csv('/content/amz_ca_total_products_data_processed.csv')

# Sample the data
df_sampled = df.sample(frac=0.1, random_state=42)

# Select relevant columns
df_sampled = df_sampled[['stars', 'reviews', 'price', 'categoryName', 'isBestSeller']]

# Drop rows with missing values
df_sampled.dropna(inplace=True)

# Convert 'isBestSeller' to boolean
df_sampled['isBestSeller'] = df_sampled['isBestSeller'].astype(bool)

# One-hot encode categorical features
encoder = OneHotEncoder()
encoded_features = encoder.fit_transform(df_sampled[['categoryName', 'isBestSeller']])
encoded_features_df = pd.DataFrame(encoded_features.toarray(), columns=encoder.get_feature_names_out())

# Combine the encoded features with the numerical features
preprocessed_df = pd.concat([encoded_features_df, df_sampled[['reviews', 'price', 'stars']].reset_index(drop=True)], axis=1)

# Split data into features and target variable
X = preprocessed_df.drop('stars', axis=1)
y = preprocessed_df['stars']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a regression model
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

# Predict on the test set
y_pred = regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
accuracy = regressor.score(X_test, y_test)

print(f'Mean Squared Error: {mse}')
print(f'Accuracy: {accuracy}')

# Save the model and encoder
joblib.dump(regressor, '/content/random_forest_regressor_model.pkl')
joblib.dump(encoder, '/content/encoder.pkl')
