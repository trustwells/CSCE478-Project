import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---- Clear Terminal ----
import os
os.system('cls' if os.name == 'nt' else 'clear')

# ---- 1. Data Loading ----
# Read the CSV file with cab_rides data
df = pd.read_csv('cab_rides.csv')

# Sample 2000 rows for analysis (using a fixed random state for reproducibility)
df = df.sample(n=2000, random_state=42)

# ---- 2. Data Preprocessing ----
# For Model 1, only using distance and price

# Check for missing values
print(f"Missing values in distance: {df['distance'].isna().sum()}")
print(f"Missing values in price: {df['price'].isna().sum()}")

# Drop rows with missing values
df = df.dropna(subset=['distance', 'price'])

# ---- 3. Exploratory Data Analysis ----
# Basic statistics
print("\nBasic Statistics:")
print(df[['distance', 'price']].describe())

# Create a scatter plot of distance vs price
plt.figure(figsize=(10, 6))
plt.scatter(df['distance'], df['price'], alpha=0.5)
plt.title('Distance vs Price')
plt.xlabel('Distance (miles)')
plt.ylabel('Price ($)')
plt.savefig('distance_vs_price.png')
plt.close()

# ---- 4. Split Data ----
X = df[['distance']]  # Feature
y = df['price']       # Target

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# ---- 5. Model Training (Linear Regression) ----
# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Print model coefficients
print("\nModel Coefficients:")
print(f"Intercept: {model.intercept_:.4f}")
print(f"Slope (distance coefficient): {model.coef_[0]:.4f}")

# ---- 6. Model Evaluation ----
# Make predictions on test set
y_pred = model.predict(X_test)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

# ---- 7. Visualize Results ----
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual Prices')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Prices')
plt.title('Linear Regression: Actual vs Predicted Prices')
plt.xlabel('Distance (miles)')
plt.ylabel('Price ($)')
plt.legend()
plt.savefig('model1_results.png')
plt.close()

# ---- 8. Sample Predictions ----
# Sample distances to test for their price
sample_distances = np.array([0.5, 1.0, 2.0, 3.0, 5.0]).reshape(-1, 1)
sample_predictions = model.predict(sample_distances)

print("\nSample Predictions:")
for distance, predicted_price in zip(sample_distances.flatten(), sample_predictions):
    print(f"Distance: {distance:.1f} miles => Predicted Price: ${predicted_price:.2f}")

# ---- 9. Model Equation ----
print("\nModel Equation:")
print(f"Price = {model.intercept_:.4f} + {model.coef_[0]:.4f} × Distance")