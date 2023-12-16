import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# Load the tokenizer and the padded sequences
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

train_padded = np.load('train_padded.npy')
test_padded = np.load('test_padded.npy')

# Assuming train_padded and test_padded are both 2D arrays of shape (num_samples, sequence_length)

# The encoder will take the padded sequences as input
encoder_input_data = train_padded

# For the decoder input, we shift the `train_padded` and `test_padded` one step to the right.
decoder_input_data = np.pad(train_padded[:, :-1], ((0, 0), (1, 0)), mode='constant', constant_values=0)
decoder_target_data = np.expand_dims(train_padded, -1)  # Targets must be 3D for sparse categorical crossentropy

# Load the model
model = tf.keras.models.load_model('hindia_spell_checker_model.keras')

# Define training parameters
batch_size = 10
epochs = 20

# Early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=1,
    restore_best_weights=True
)

# Train the model with early stopping
history = model.fit(
    [encoder_input_data, decoder_input_data],  # Encoder and decoder input
    decoder_target_data,                       # Decoder target
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(
        [test_padded, np.pad(test_padded[:, :-1], ((0, 0), (1, 0)), mode='constant', constant_values=0)],
        np.expand_dims(test_padded, -1)  # Validation data
    ),
    callbacks=[early_stopping]
)

# Save the trained model
model.save('hindia_spell_checker_trained_model.keras')
