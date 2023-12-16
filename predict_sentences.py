import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the trained model
model = tf.keras.models.load_model('hindia_spell_checker_trained_model.keras')

# Load the maximum sequence length
with open('max_sequence_len.txt', 'r') as f:
    max_sequence_len = int(f.read().strip())

# Function to preprocess new text data for prediction
def preprocess_text(sentences, tokenizer, max_sequence_len):
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_len, padding='post')
    return padded_sequences

# Dummy data for prediction
new_sentences = ['??????', '???', '????','????','??????','????','????','????']

# Preprocess the text
new_padded_sequences = preprocess_text(new_sentences, tokenizer, max_sequence_len)

# Since we are not using a `<start>` token, we'll initialize the decoder input as an array of zeros
decoder_input_data = np.zeros((new_padded_sequences.shape[0], max_sequence_len), dtype=np.int32)

# Predict using the model
predictions = model.predict([new_padded_sequences, decoder_input_data])

# Convert predictions to text
for i, sequence in enumerate(predictions):
    sequence_text = []
    for word_probabilities in sequence:
        chosen_word_idx = np.argmax(word_probabilities)
        word = tokenizer.index_word.get(chosen_word_idx, 'UNK')
        if word != 'UNK':
            sequence_text.append(word)
    print('Predicted sentence:', ' '.join(sequence_text))
