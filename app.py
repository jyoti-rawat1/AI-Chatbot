import streamlit as st
import numpy as np
import json
import pickle
import tensorflow as tf

# ---------------- Load Model & Files ---------------- #

model = tf.keras.models.load_model("model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    lbl_encoder = pickle.load(f)

with open("intents.json") as file:
    data = json.load(file)

max_len = 50

# ---------------- Prediction Function ---------------- #

def predict_intent(user_input):
    sequence = tokenizer.texts_to_sequences([user_input.lower()])
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        sequence, maxlen=max_len, padding="post"
    )

    prediction = model.predict(padded)
    confidence = np.max(prediction)
    predicted_index = np.argmax(prediction)
    predicted_tag = lbl_encoder.inverse_transform([predicted_index])[0]

    return predicted_tag, confidence


def get_response(tag):
    for intent in data["intents"]:
        if intent["tag"] == tag:
            return np.random.choice(intent["responses"])

    return "Sorry, I didn't understand that."


# ---------------- Streamlit UI ---------------- #

st.set_page_config(page_title="Career Guidance Chatbot")
st.title("🎓 Career Guidance Chatbot")
st.write("Ask me anything about your career roadmap!")

# Chat history storage
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        st.chat_message("assistant").write(message["content"])

# User input box
user_input = st.chat_input("Type your question here...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Predict
    tag, confidence = predict_intent(user_input)

    if confidence < 0.4:
        response = "I'm not fully sure. Can you please rephrase your question?"
    else:
        response = get_response(tag)

    # Show bot response
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)