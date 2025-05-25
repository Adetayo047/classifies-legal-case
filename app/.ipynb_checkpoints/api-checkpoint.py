from model_utils import load_model, load_tokenizer, get_max_len, clean, label_map
from flask import Flask, request, jsonify
import joblib
import pickle
import re
import string
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load model and tokenizer
model = load_model()
tokenizer = load_tokenizer()
maxlen = get_max_len()

@app.route('/pred_label', methods=['POST'])
def predict_area_of_law():
    data = request.get_json()
    if not data or 'full_report' not in data:
        return jsonify({'error': 'Please provide full_report text'}), 400

    full_report = data['full_report']
    cleaned = clean(full_report)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded_seq = pad_sequences(seq, maxlen=maxlen)

    # Predict label index
    pred_encoded = model.predict(padded_seq.reshape(1, -1))
    pred_label = label_map.get(pred_encoded[0], "Unknown")
    print("Received data:", data)

    return jsonify({'pred_label': pred_label})

if __name__ == '__main__':
    app.run(debug = True, port=5000)
