import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('hindia_spell_checker_trained_model.keras')

# Load the test data (make sure the paths are correct)
test_padded = np.load('test_padded.npy')

# Prepare the test data input as done for the training data
test_decoder_input = np.pad(test_padded[:, :-1], ((0, 0), (1, 0)), mode='constant', constant_values=0)

# Evaluate the model
loss, accuracy = model.evaluate([test_padded, test_decoder_input], np.expand_dims(test_padded, -1))
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')
