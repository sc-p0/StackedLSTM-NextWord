import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

st.set_page_config(page_title="Feynman LSTM Generator", page_icon="⚡")

st.title("Richard Feynman Memoir Next-Word Generator")
st.markdown("A **Stacked LSTM** neural network trained on Richard Feynman's autobiographical writings.")

# Load cached model and tokenizer  
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('feynman_model.h5')
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_assets()

# UI Layout
input_text = st.text_input("Starting Phrase:", "My experiments were always being")
num_words = st.slider("Number of words to generate:", 1, 20, 10)

if st.button("Generate Completion"):
    current_text = input_text
    
    with st.spinner("Generating text..."):
        for _ in range(num_words):
            token_text = tokenizer.texts_to_sequences([current_text])[0]
            padded_token_text = pad_sequences([token_text], maxlen=93, padding='pre')
            pos = np.argmax(model.predict(padded_token_text, verbose=0))
            
            next_word = ""
            for word, index in tokenizer.word_index.items():
                if index == pos:
                    next_word = word
                    break
            
            if next_word:
                current_text += " " + next_word
            else:
                break
                
    st.markdown("### Result:")
    st.info(current_text)