import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Flatten

# Sample data (replace this with your preprocessed dataset)
# X_train and y_train should be numpy arrays with appropriate shapes
X_train = np.random.randn(
    100, 10, 1
)  # Sample input data with shape (samples, timesteps, features)
y_train = np.random.randint(1, 10, size=(100,))  # Sample target data (length of stay)


# Define the hybrid model
def cnn_lstm_model(input_shape):
    inputs = Input(shape=input_shape)
    cnn_layer1 = Conv1D(filters=32, kernel_size=3, activation="relu")(inputs)
    cnn_layer1 = MaxPooling1D(pool_size=2)(cnn_layer1)
    cnn_layer2 = Conv1D(filters=64, kernel_size=3, activation="relu")(cnn_layer1)
    cnn_layer2 = MaxPooling1D(pool_size=2)(cnn_layer2)
    cnn_layer3 = Conv1D(filters=128, kernel_size=3, activation="relu")(cnn_layer2)
    cnn_layer3 = MaxPooling1D(pool_size=2)(cnn_layer3)
    flatten_layer = Flatten()(cnn_layer3)
    lstm_layer = LSTM(units=64)(flatten_layer)
    outputs = Dense(units=1)(
        lstm_layer
    )  # Output layer with one neuron for regression task
    model = Model(inputs=inputs, outputs=outputs)
    return model


# Create the model
model = cnn_lstm_model(input_shape=X_train.shape[1:])

# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error")

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Once trained, you can use the model for predictions
# Assuming X_test contains your test data with the same shape as X_train
# Replace X_test with your actual test data
X_test = np.random.randn(
    10, 10, 1
)  # Sample test data with shape (samples, timesteps, features)
predictions = model.predict(X_test)

# Print predicted length of stay
print(predictions)
