import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------
# LOAD MODEL + TOKENIZER
# -------------------------
@st.cache_resource
def load_all():
    model = load_model("next_word_model.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_all()

max_len = 18  # your value

# -------------------------
# PREDICTION FUNCTION
# -------------------------
def predict_next_word(text):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
    
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted)
    
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word

# -------------------------
# UI
# -------------------------
st.title("Next Word Prediction App")
st.write("Type a sentence and get the next word prediction")

user_input = st.text_input("Enter your text:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        next_word = predict_next_word(user_input)
        st.success(f"Next word: {next_word}")