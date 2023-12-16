import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# Load the data
train_data_path = 'train_data.csv'  # Make sure the path is correct
test_data_path = 'test_data.csv'    # Make sure the path is correct

train_data = pd.read_csv(train_data_path, header=None, names=['sentence'])
test_data = pd.read_csv(test_data_path, header=None, names=['sentence'])

# Ensure all data is string type and not missing
train_data['sentence'] = train_data['sentence'].fillna('').astype(str)
test_data['sentence'] = test_data['sentence'].fillna('').astype(str)

# Define the tokenizer
tokenizer = Tokenizer(char_level=False)  # Use `char_level=True` if the tokenization should be at the character level
tokenizer.fit_on_texts(train_data['sentence'])

# Convert sentences to sequences of integers
train_sequences = tokenizer.texts_to_sequences(train_data['sentence'])
test_sequences = tokenizer.texts_to_sequences(test_data['sentence'])

# Pad sequences to have the same length
max_sequence_len = max(max(len(seq) for seq in train_sequences), max(len(seq) for seq in test_sequences))
train_padded = pad_sequences(train_sequences, maxlen=max_sequence_len, padding='post')
test_padded = pad_sequences(test_sequences, maxlen=max_sequence_len, padding='post')

# Save the tokenizer and the padded sequences for later use
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save the padded sequences as numpy files for efficient loading
np.save('train_padded.npy', train_padded)
np.save('test_padded.npy', test_padded)

# Optionally, also save the maximum sequence length
with open('max_sequence_len.txt', 'w') as f:
    f.write(str(max_sequence_len))
