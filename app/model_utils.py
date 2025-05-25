import joblib
import pickle
import os
import html
import re
import string
import nltk
from nltk.corpus import stopwords

# Define paths to saved model and tokenizer 
MODEL_PATH = os.path.join('models/DT_model.pkl')
TOKENIZER_PATH = os.path.join('models/vect.pkl')
MAXLEN = os.path.join('models/max_len.pkl')

def load_model():
    """
    Load and return the trained model.
    """
    model = joblib.load(MODEL_PATH)
    return model

def load_tokenizer():
    """
    Load and return the fitted tokenizer.
    """
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

def get_max_len():
    """
    Load and return the fitted tokenizer.
    """
    with open(MAXLEN, 'rb') as f:
        maxlen = pickle.load(f)
    return maxlen


# function to clean input text. 
def clean(text):

  # remove non-breaking space
  text = ' '.join(html.unescape(word).replace("\xa0", " ") for word in text.split())

  # save the pattern for the punctuation and special characters in regex
  re_punc = re.compile('[%s]' % re.escape(string.punctuation))

  # replace the punctuation with nothing for each word in the list
  # and join the words back into a string
  text = ' '.join([re_punc.sub('', word) for word in text.split()])

  # remove extra special characters not handled above
  text = ' '.join([re.sub(r'[’,-, @]','', word) for word in text.split()])

  # remove non-alphabets
  text = ' '.join([word for word in text.split() if word.isalpha()])

  # filter out stop words
  stop_words = set(stopwords.words('english'))
  text = ' '.join([word for word in text.split() if not word in stop_words])

  #remove extra whitspace
  text = text.strip()

  # lower case
  text = text.lower()

  return text

# Define the mapping dictionary
label_map = {
    0: "Civil Procedure",
    1: "Criminal Law and Procedure",
    2: "Election Petition",
    3: "Enforcement of Fundamental right",
    4: "Land Law",
    5: "Other"
}
