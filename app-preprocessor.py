import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer
with open("app/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

def preprocess_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=20)
    return padded
