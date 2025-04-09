import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 1. Data Loading
def load_data(cab_rides_path, weather_path, sample_size=20000):
    """
    Load the cab rides and weather datasets with sampling
    """
    print("Loading datasets...")
    
    # Load datasets
    cab_rides = pd.read_csv(cab_rides_path)
    weather = pd.read_csv(weather_path)
    
    print(f"Original cab rides shape: {cab_rides.shape}")
    print(f"Weather shape: {weather.shape}")
    
    # Sample only 20,000 records from cab_rides
    if len(cab_rides) > sample_size:
        print(f"Sampling {sample_size} records from {len(cab_rides)} cab rides")
        cab_rides = cab_rides.sample(sample_size, random_state=42)
    
    print(f"Sampled cab rides shape: {cab_rides.shape}")
    
    # Display sample of the data
    print("\nCab rides sample:")
    print(cab_rides.head())
    
    print("\nWeather sample:")
    print(weather.head())
    
    return cab_rides, weather

# 2. Basic Preprocessing
def basic_preprocess(cab_rides, weather):
    """
    Perform only basic preprocessing - convert timestamps and handle missing values
    """
    print("\nPerforming basic preprocessing...")
    
    # Convert timestamps to datetime
    cab_rides['time_stamp'] = pd.to_datetime(cab_rides['time_stamp'], unit='ms')
    weather['time_stamp'] = pd.to_datetime(weather['time_stamp'], unit='s')
    
    # Handle missing values in cab_rides - just drop rows with missing price
    cab_rides = cab_rides.dropna(subset=['price'])
    
    # Check for any invalid surge multipliers and fix them (without feature engineering)
    if 'surge_multiplier' in cab_rides.columns:
        # Convert to numeric if needed
        cab_rides['surge_multiplier'] = pd.to_numeric(cab_rides['surge_multiplier'], errors='coerce')
        # Replace missing values with 1.0
        cab_rides['surge_multiplier'] = cab_rides['surge_multiplier'].fillna(1.0)
    
    print(f"Processed cab rides shape: {cab_rides.shape}")
    
    # Display column information
    print("\nCab rides columns:")
    print(cab_rides.columns.tolist())
    
    print("\nWeather columns:")
    print(weather.columns.tolist())
    
    # Check for missing values
    print("\nMissing values in cab_rides:")
    print(cab_rides.isnull().sum())
    
    print("\nMissing values in weather:")
    print(weather.isnull().sum())
    
    return cab_rides, weather

# 3. Save Processed Data
def save_processed_data(cab_rides, weather):
    """
    Save the processed data to CSV files
    """
    print("\nSaving processed data...")
    
    cab_rides.to_csv('processed_cab_rides_20k.csv', index=False)
    weather.to_csv('processed_weather.csv', index=False)
    
    print("Data saved successfully!")

# 4. Main function
def main():
    """
    Main function to load and minimally process the data
    """
    print("Starting Minimal Data Processing for Uber & Lyft Data...")
    
    # 1. Load data with 20,000 sample size
    cab_rides, weather = load_data('cab_rides.csv', 'weather.csv', sample_size=20000)
    
    # 2. Basic preprocessing
    processed_cab_rides, processed_weather = basic_preprocess(cab_rides, weather)
    
    # 3. Save processed data
    save_processed_data(processed_cab_rides, processed_weather)
    
    print("\nMinimal data processing completed successfully!")
    
    return processed_cab_rides, processed_weather

if __name__ == "__main__":
    processed_cab_rides, processed_weather = main()