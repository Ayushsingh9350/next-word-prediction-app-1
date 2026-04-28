import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------
# LOAD MODEL
# -------------------------
model = load_model("next_word_model.keras")

# -------------------------
# LOAD TOKENIZER
# -------------------------
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# -------------------------
# SET MAX LENGTH
# -------------------------
max_len = 18

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
# RUN LOOP
# -------------------------
print("Type something (type 'exit' to quit)\n")

while True:
    text = input("Input: ")
    
    if text.lower() == "exit":
        break
    
    next_word = predict_next_word(text)
    print("Next word:", next_word)