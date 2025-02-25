import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Step 1: Load the Data
# Assuming data comes from a CSV with columns: 'timestamp', 'glucose_level', 'basal_insulin', 'bolus_insulin', 'carbs'
data = pd.read_csv('glucose_data.csv', parse_dates=['timestamp'], index_col='timestamp')

# Step 2: Preprocess the Data
# Handle missing values
data['glucose_level'].fillna(method='ffill', inplace=True)  # Forward fill glucose levels
data[['basal_insulin', 'bolus_insulin', 'carbs']] = data[['basal_insulin', 'bolus_insulin', 'carbs']].fillna(0)  # Fill insulin/carbs with 0

# Define features and target
features = ['glucose_level', 'basal_insulin', 'bolus_insulin', 'carbs']
target = 'glucose_level'

# Scale all features to [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
data[features] = scaler.fit_transform(data[features])

# Store glucose min/max for inverse scaling later
glucose_min = scaler.data_min_[0]  # Glucose is the first feature
glucose_max = scaler.data_max_[0]

# Function to create sequences
def create_sequences(data, n_steps, prediction_horizon, features, target):
    """
    Create sequences for LSTM with multiple features.
    
    Parameters:
    - data: DataFrame with scaled features
    - n_steps: Number of past time steps (e.g., 12 for 60 minutes)
    - prediction_horizon: Steps ahead to predict (e.g., 12 for 60 minutes)
    - features: List of input feature columns
    - target: Column to predict
    
    Returns:
    - X: Input sequences (shape: [samples, n_steps, n_features])
    - y: Target values (shape: [samples])
    """
    X, y = [], []
    for i in range(n_steps, len(data) - prediction_horizon + 1):
        X.append(data[features].iloc[i - n_steps:i].values)  # Past n_steps with all features
        y.append(data[target].iloc[i + prediction_horizon - 1])  # Glucose value prediction_horizon steps ahead
    return np.array(X), np.array(y)

# Set parameters: 60 minutes of past data, predict 60 minutes ahead
n_steps = 12  # 12 steps = 60 minutes with 5-minute intervals
prediction_horizon = 12  # Predict 60 minutes into the future
X, y = create_sequences(data, n_steps, prediction_horizon, features, target)

# Split into train and test sets (no shuffling to preserve time order)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 3: Build the LSTM Model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(n_steps, len(features))),  # Input shape: (timesteps, features)
    Dropout(0.2),  # Prevent overfitting
    Dense(1)  # Output a single glucose prediction
])
model.compile(optimizer='adam', loss='mse')

# Step 4: Train the Model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Step 5: Evaluate the Model
y_pred = model.predict(X_test)

# Inverse scale the predictions and actual values
y_test_rescaled = y_test * (glucose_max - glucose_min) + glucose_min
y_pred_rescaled = y_pred * (glucose_max - glucose_min) + glucose_min

# Calculate Root Mean Squared Error
rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test_rescaled, label='Actual Glucose Levels')
plt.plot(y_pred_rescaled, label='Predicted Glucose Levels')
plt.title('Glucose Level Prediction (60 Minutes Ahead)')
plt.xlabel('Time Steps')
plt.ylabel('Glucose Level')
plt.legend()
plt.show()

# Step 6: Predict Glucose One Hour into the Future
last_sequence = data[features].iloc[-n_steps:].values.reshape(1, n_steps, len(features))
future_prediction = model.predict(last_sequence)
future_prediction_rescaled = future_prediction * (glucose_max - glucose_min) + glucose_min
print(f'Predicted glucose level one hour into the future: {future_prediction_rescaled[0][0]:.2f}')