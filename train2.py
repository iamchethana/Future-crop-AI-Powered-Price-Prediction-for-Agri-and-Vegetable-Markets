# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score

# Load and preprocess the data
data = pd.read_excel('Vegetable and Fruits Prices  in India.xlsx')

# Data Cleaning and Feature Engineering
data.drop(columns=['datesk'], inplace=True)
data = data[~data['Item Name'].isnull()]
data['Date'] = pd.to_datetime(data['Date'])
data['year'] = data['Date'].dt.year
data.drop(columns='Date', inplace=True)

# Remove rows where price is null or price is 0
data = data[~((data['price'].isnull()) | (data['price'] == 0))]

# Log transform to normalize price data
data['price'] = np.log(data['price'])

# Preparing the training and test sets
train_data = pd.get_dummies(data.drop(columns=['price']), drop_first=True)
train_output = data['price']
X_train, X_test, y_train, y_test = train_test_split(train_data, train_output, test_size=0.2, random_state=42)

# Scale the data for neural network
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define a simple neural network model
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer for regression
])

# Compile the model with loss and optimizer
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_squared_error'])

# Train the model and store the history
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, verbose=1)

# Predict on train and test data to calculate R-squared (as a measure of accuracy for regression)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate R-squared for training and testing data
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Print training and testing accuracy (R-squared)
print(f"Training R-squared: {train_r2:.4f}")
print(f"Testing R-squared: {test_r2:.4f}")

# Plotting training & validation loss values
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()

# R-squared values graph to show "accuracy"
plt.subplot(1, 2, 2)
epochs = range(1, len(history.history['loss']) + 1)
train_r2_list = [train_r2] * len(epochs)  # Constant R-squared
test_r2_list = [test_r2] * len(epochs)    # Constant R-squared
plt.plot(epochs, train_r2_list, label=f'Training R-squared = {train_r2:.4f}')
plt.plot(epochs, test_r2_list, label=f'Testing R-squared = {test_r2:.4f}')
plt.title('Model R-squared (Accuracy)')
plt.xlabel('Epoch')
plt.ylabel('R-squared')
plt.legend()

plt.show()
