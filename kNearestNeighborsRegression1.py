import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---- Clear Terminal ----
import os
os.system('cls' if os.name == 'nt' else 'clear')

# ---- 1. Data Loading ----
# Read the CSV file
df = pd.read_csv('cab_rides.csv')

# Get rows for analysis
df = df.sample(n=10000, random_state=69)

# ---- 2. Data Preprocessing ----
# Remove instances with missing values
df = df.dropna(subset=['distance', 'price', 'product_id'])

# Process cab type from product_id
def extract_cab_type(product_id):
    # Check if it's a UUID (driver ID)
    if '-' in str(product_id):
        return 'standard'  # Default for UUID values
    
    # Get the service type from product_id (ride type)
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
        return 'standard'  # Default for basic ride

# Use function to create the new column
df['cab_type'] = df['product_id'].apply(extract_cab_type)

# ---- 3. Feature Selection and Data Splitting ----
# Define features and target
X = df[['distance', 'cab_type']]
y = df['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# ---- 4. Create Pipeline with Preprocessing and Fixed Parameters ----
# Define preprocessing for numeric and categorical features
numeric_features = ['distance']
categorical_features = ['cab_type']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create pipeline with optimal parameters (K=15, manhattan distance, uniform weights)
knn_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('knn', KNeighborsRegressor(n_neighbors=15, p=1, weights='uniform'))
])

# ---- 5. Model Training ----
print("Training KNN model with optimal parameters...")
knn_pipeline.fit(X_train, y_train)

# ---- 6. Model Evaluation ----
test_predictions = knn_pipeline.predict(X_test)

# Calculate performance metrics
test_mae = mean_absolute_error(y_test, test_predictions)
test_mse = mean_squared_error(y_test, test_predictions)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, test_predictions)

print("\nTest Set Performance:")
print(f"Mean Absolute Error (MAE): {test_mae:.4f}")
print(f"Mean Squared Error (MSE): {test_mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {test_rmse:.4f}")
print(f"R-squared (R²): {test_r2:.4f}")

# ---- 7. Visualize Results ----
plt.figure(figsize=(10, 6))
plt.scatter(X_test['distance'], y_test, color='blue', alpha=0.5, label='Actual Prices')
plt.scatter(X_test['distance'], test_predictions, color='green', alpha=0.5, 
           label=f'KNN Model (R²={test_r2:.3f})')
plt.title('KNN Regression: Actual vs Predicted Prices')
plt.xlabel('Distance (miles)')
plt.ylabel('Price ($)')
plt.legend()
plt.tight_layout()
plt.savefig('knn_model_results.png') # What the output is saved as
plt.close()

# ---- 8. Sample Predictions ----
def predict_price(distance, cab_type):
    sample = pd.DataFrame({
        'distance': [distance],
        'cab_type': [cab_type]
    })
    
    return knn_pipeline.predict(sample)[0]

print("\nSample Predictions for this Model:")
for distance, cab_type in [(1.0, 'economy'), (1.0, 'premium'), (3.0, 'standard')]:
    price = predict_price(distance, cab_type)
    print(f"Distance: {distance:.1f} miles, Cab Type: {cab_type} => Predicted Price: ${price:.2f}")
