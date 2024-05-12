import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Mean

# Sample data (replace this with your preprocessed dataset)
# X_train and y_train should be numpy arrays with appropriate shapes
X_train = np.random.randn(
    100, 10, 1
)  # Sample input data with shape (samples, timesteps, features)
y_train = np.random.randint(1, 10, size=(100,))  # Sample target data (length of stay)


# Define Transformer model
def transformer_model(input_shape):
    inputs = Input(shape=input_shape)
    transformer_block = tf.keras.layers.Transformer(
        num_layers=2,  # Number of Transformer layers
        d_model=64,  # Dimensionality of the model
        num_heads=4,  # Number of attention heads
        activation="relu",
        dropout=0.1,
        name="transformer",
    )(inputs)
    transformer_block = LayerNormalization(epsilon=1e-6)(transformer_block)
    transformer_block = Flatten()(transformer_block)
    outputs = Dense(1)(
        transformer_block
    )  # Output layer with one neuron for regression task
    model = Model(inputs=inputs, outputs=outputs)
    return model


# Create Transformer model
model = transformer_model(input_shape=X_train.shape[1:])

# Compile the model
model.compile(optimizer=Adam(), loss=MeanSquaredError(), metrics=[MeanSquaredError()])

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
