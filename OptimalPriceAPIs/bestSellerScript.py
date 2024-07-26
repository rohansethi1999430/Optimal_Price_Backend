import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Assuming the dataset is loaded in a DataFrame called df
# df = pd.read_csv('path_to_your_dataset.csv')  # Uncomment and use the correct path

# Sample data preparation (mockup based on the provided structure)
data = {
    "title": ["Product A", "Product B", "Product C"],
    "categoryName": ["Electronics", "Baby", "Electronics"],
    "stars": [4.5, 3.0, 2.0],
    "reviews": [150, 10, 5],
    "price": [49.99, 25.00, 10.00],
    "isBestSeller": [True, False, False]
}

df = pd.DataFrame(data)

# Preprocessing
# Convert 'isBestSeller' to binary
df['isBestSeller'] = df['isBestSeller'].astype(int)

# Encode categorical variables
df = pd.get_dummies(df, columns=['categoryName'])

# Define features and target
X = df.drop(columns=['title', 'isBestSeller'])
y = df['isBestSeller']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model and the feature names
dump(model, 'bestseller_predictor.joblib')
dump(X_train.columns.tolist(), 'bestseller_predictor_features.pkl')
