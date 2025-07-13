import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

df = pd.read_csv("app/sample_data.csv")

# Encode labels
le = LabelEncoder()
df["label_enc"] = le.fit_transform(df["label"])

# Save the label encoder
with open("app/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Tokenization
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(df["text"])
sequences = tokenizer.texts_to_sequences(df["text"])
X = pad_sequences(sequences, maxlen=20)
y = df["label_enc"]

# Save tokenizer
with open("app/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=1000, output_dim=16, input_length=20),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(len(df["label_enc"].unique()), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=10)

# Save model
model.save("app/medical_model.h5")
