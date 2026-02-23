import json
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# ---------------- Load intents.json ---------------- #

with open("intents.json") as file:
    data = json.load(file)

training_sentences = []
training_labels = []
labels = []
responses = {}

# Extract patterns and tags
for intent in data["intents"]:
    tag = intent["tag"]
    labels.append(tag)
    responses[tag] = intent["responses"]
    
    for pattern in intent["patterns"]:
        training_sentences.append(pattern.lower())
        training_labels.append(tag)

# ---------------- Label Encoding ---------------- #

lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels_encoded = lbl_encoder.transform(training_labels)

num_classes = len(set(training_labels_encoded))
training_labels_categorical = to_categorical(training_labels_encoded)

# ---------------- Tokenization ---------------- #

vocab_size = 5000
max_len = 50
embedding_dim = 128

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(training_sentences)

sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

# ---------------- LSTM Model ---------------- #

model = Sequential()

model.add(Embedding(vocab_size, 128, input_length=max_len))

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.5))

model.add(LSTM(64))
model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# ---------------- Train Model ---------------- #

model.fit(
    padded_sequences,
    training_labels_categorical,
    epochs=500,
    verbose=1
)

# ---------------- Save Model & Files ---------------- #

model.save("model.h5")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(lbl_encoder, f)

print("✅ Model Training Complete!")
print("✅ model.h5, tokenizer.pkl, label_encoder.pkl saved successfully!")