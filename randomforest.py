import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from math import sqrt
import joblib

# 1. Load the dataset
file_path = r'D:\Python\Random Forest\ZENODO\3.cleaned_dataset_per_device.csv'
data = pd.read_csv(file_path)

# Define features (X) and target (y)

features = [
    'distance',
    'c_walls',
    'w_walls',
    'co2',
    'humidity',
    'pm25',
    'pressure',
    'temperature',
    'frequency',  # Frequency used in the transmission
    'rssi',  # Received signal strength
    'snr'  # Signal-to-noise ratio
]


target = 'SF'

X = data[features]
y = data[target]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# 5. Make predictions and evaluate performance
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Random Forest RMSE: {rmse:.2f}")
print(f"Random Forest R²: {r2:.2f}")

# 6. Save the trained model for future use
model_path = r'D:\Python\Random Forest\ZENODO\random_forest_sf_model.pkl'
joblib.dump(rf_model, model_path)
print(f" Model saved to {model_path}")


# 7. Function to load model and predict SF for new input
def predict_sf(new_input):
    # Load the trained model
    loaded_model = joblib.load(model_path)

    # Convert new_input dict to DataFrame
    input_df = pd.DataFrame([new_input])

    # Predict SF
    predicted_spreading_factor = loaded_model.predict(input_df)[0]

    return predicted_spreading_factor


# Example: Predict SF for new sensor data
new_sensor_data = {
    'distance': 15,
    'c_walls': 2,
    'w_walls': 1,
    # 'f_count': 77,
    # 'p_count': 113,
    'co2': 550,
    'humidity': 45.0,
    'pm25': 0.5,
    'pressure': 1013.25,
    'temperature': 22.5,
    'frequency': 900,
    'rssi': -85,
    'snr': 10.5,
}

predicted_sf = predict_sf(new_sensor_data)
print(f"Predicted Spreading Factor (SF): {predicted_sf:.1f}")
