# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

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
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_absolute_error'])

# Train the model and store the history
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

# Plotting training & validation loss values
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plotting training & validation accuracy values
plt.subplot(1, 2, 2)
plt.plot(history.history['mean_absolute_error'], label='Training MAE')
plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('Model Accuracy (MAE)')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()

plt.show()
