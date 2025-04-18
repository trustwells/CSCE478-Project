import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---- Clear Terminal ----
import os
os.system('cls' if os.name == 'nt' else 'clear')

# ---- 1. Data Loading ----
# Read the CSV file with cab_rides data
df = pd.read_csv('cab_rides.csv')

# Get rows for analysis
df = df.sample(n=10000, random_state=123)

# ---- 2. Data Preprocessing ----
print("Checking columns in dataset:")
print(df.columns.tolist())

# Check for missing values in key columns
print("\nMissing values in key columns:")
for col in ['distance', 'price', 'product_id']:
    print(f"{col}: {df[col].isna().sum()}")

# Handle missing values
df = df.dropna(subset=['distance', 'price', 'product_id'])

# Process cab type from product_id - extract the service level
def extract_cab_type(product_id):
    # Check if it's a UUID (driver ID)
    if '-' in str(product_id):
        return 'standard'  # Default for UUID values
    
    # Extract the service type from product_id strings
    prod_id = str(product_id).lower()
    
    if any(x in prod_id for x in ['line', 'pool', 'shared']):
        return 'economy'
    elif any(x in prod_id for x in ['xl', 'plus', 'wav']):
        return 'xl'
    elif any(x in prod_id for x in ['black', 'suv', 'luxsuv']):
        return 'premium'
    elif any(x in prod_id for x in ['lux', 'premier', 'select']):
        return 'luxury'
    else:
        return 'standard'  # Default for basic Uber/Lyft

# Apply the function to create a new column
df['cab_type'] = df['product_id'].apply(extract_cab_type)

# ---- 3. Exploratory Data Analysis ----
# Basic statistics
print("\nBasic Statistics:")
print(df[['distance', 'price']].describe())

# Frequency of different cab types
print("\nCab Type Distribution:")
print(df['cab_type'].value_counts())

# Average price by cab type
print("\nAverage Price by Cab Type:")
print(df.groupby('cab_type')['price'].mean().sort_values(ascending=False))

# ---- 4. Feature Selection and Data Splitting ----
# Define features and target
X_numeric = df[['distance']]
X_categorical = df[['cab_type']]
y = df['price']

# Split data into training (80%) and testing (20%) sets
X_numeric_train, X_numeric_test, X_categorical_train, X_categorical_test, y_train, y_test = train_test_split(
    X_numeric, X_categorical, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {len(X_numeric_train)} samples")
print(f"Testing set size: {len(X_numeric_test)} samples")

# ---- 5. Create Pipeline with Preprocessing ----
numeric_features = ['distance']
categorical_features = ['cab_type']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('knn', KNeighborsRegressor())
])

# ---- 6. Model Tuning with GridSearchCV ----
param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9, 11, 15],
    'knn__weights': ['uniform', 'distance'],
    'knn__p': [1, 2]  # p=1 for Manhattan, p=2 for Euclidean
}

X_train = pd.concat([X_numeric_train.reset_index(drop=True), 
                    X_categorical_train.reset_index(drop=True)], axis=1)
X_test = pd.concat([X_numeric_test.reset_index(drop=True), 
                   X_categorical_test.reset_index(drop=True)], axis=1)

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters:", grid_search.best_params_)
print(f"Best Cross-Validation Score: {-grid_search.best_score_:.4f} (MSE)")

# ---- 7. Final Evaluation on Test Set ----
best_model = grid_search.best_estimator_
test_predictions = best_model.predict(X_test)

test_mae = mean_absolute_error(y_test, test_predictions)
test_mse = mean_squared_error(y_test, test_predictions)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, test_predictions)

print("\nTest Set Performance:")
print(f"Mean Absolute Error (MAE): {test_mae:.4f}")
print(f"Mean Squared Error (MSE): {test_mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {test_rmse:.4f}")
print(f"R-squared (R²): {test_r2:.4f}")

# ---- 8. Visualize Results ----
plt.figure(figsize=(10, 6))
plt.scatter(X_test['distance'], y_test, color='blue', alpha=0.5, label='Actual Prices')
plt.scatter(X_test['distance'], test_predictions, color='green', alpha=0.5, 
           label=f'KNN Model (R²={test_r2:.3f})')
plt.title('KNN Regression: Actual vs Predicted Prices')
plt.xlabel('Distance (miles)')
plt.ylabel('Price ($)')
plt.legend()
plt.tight_layout()
plt.savefig('model_performance.png')
plt.close()

# ---- 9. Feature Importance Analysis ----
# Test each feature individually to understand their importance
distance_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsRegressor(
        n_neighbors=grid_search.best_params_['knn__n_neighbors'],
        weights=grid_search.best_params_['knn__weights'],
        p=grid_search.best_params_['knn__p']
    ))
])
distance_pipeline.fit(X_train[['distance']], y_train)
distance_pred = distance_pipeline.predict(X_test[['distance']])
distance_r2 = r2_score(y_test, distance_pred)

# Test categorical feature separately
cab_type_pipeline = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore')),
    ('knn', KNeighborsRegressor(
        n_neighbors=grid_search.best_params_['knn__n_neighbors'],
        weights=grid_search.best_params_['knn__weights'],
        p=grid_search.best_params_['knn__p']
    ))
])
cab_type_pipeline.fit(X_train[['cab_type']], y_train)
cab_type_pred = cab_type_pipeline.predict(X_test[['cab_type']])
cab_type_r2 = r2_score(y_test, cab_type_pred)

# Combined model R² (from the full model)
combined_r2 = test_r2

print("\nFeature Importance Analysis (R² scores):")
print(f"Distance Only: {distance_r2:.4f}")
print(f"Cab Type Only: {cab_type_r2:.4f}")
print(f"Combined Model: {combined_r2:.4f}")

# ---- 10. Sample Predictions ----
def predict_price(distance, cab_type):
    sample = pd.DataFrame({
        'distance': [distance],
        'cab_type': [cab_type]
    })
    
    return best_model.predict(sample)[0]

print("\nSample Predictions:")

scenarios = [
    # (distance, cab_type)
    (1.0, 'economy'),
    (1.0, 'standard'),
    (1.0, 'premium'),
    (1.0, 'luxury'),
    (1.0, 'xl'),
    (3.0, 'economy'),
    (3.0, 'premium')
]

for distance, cab_type in scenarios:
    price = predict_price(distance, cab_type)
    print(f"Distance: {distance:.1f} miles, Cab Type: {cab_type} => Predicted Price: ${price:.2f}")

# ---- 11. Model Description ----
k_value = grid_search.best_params_['knn__n_neighbors']
weight_type = grid_search.best_params_['knn__weights']
distance_metric = "Manhattan" if grid_search.best_params_['knn__p'] == 1 else "Euclidean"

print("\nFinal Model Description:")
print(f"K-Nearest Neighbors Regression (K={k_value}, weights={weight_type}, metric={distance_metric})")
print("Features: distance and cab_type (one-hot encoded)")
print("Preprocessing: StandardScaler for distance, OneHotEncoder for cab_type")
print(f"Selected through GridSearchCV with 5-fold cross-validation")
