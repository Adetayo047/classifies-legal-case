import streamlit as st
from model_utils import load_model, load_tokenizer, get_max_len, clean, label_map
from keras.preprocessing.sequence import pad_sequences

# Initialize model, tokenizer, and maxlen in session state once
if 'model' not in st.session_state:
    st.session_state.model = load_model()
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = load_tokenizer()
if 'maxlen' not in st.session_state:
    st.session_state.maxlen = get_max_len()

st.title("Legal Documents Predictor")
st.header("Enter the full report of a legal case and get the area of law it concerns.")

full_report = st.text_area("Enter full report text here:")

if st.button("Predict"):
    if not full_report.strip():
        st.error("Please provide full_report text")
    else:
        cleaned = clean(full_report)
        seq = st.session_state.tokenizer.texts_to_sequences([cleaned])
        padded_seq = pad_sequences(seq, maxlen=st.session_state.maxlen)

        pred_encoded = st.session_state.model.predict(padded_seq.reshape(1, -1))
    
        pred_label = label_map.get(pred_encoded[0], "Unknown")

        st.success(f"Predicted Area of Law is: **{pred_label}**")
