import streamlit as st
import pickle
import warnings
from sklearn.exceptions import InconsistentVersionWarning
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

with open("D:/Data Science/svm_model.pkl", "rb") as f:
    loaded = pickle.load(f)

with open("D:/Data Science/tfidf.pkl", "rb") as f:
    loaded_tfidf = pickle.load(f)

def cleaning(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Change all text to lower case
    text = text.lower()
        
    # Remove all special symbols
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
        
    # Perform tokenization
    tokens = word_tokenize(text)
        
    # Perform stopwords removal
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
        
    # Combine the results into a single string
    cleaned_text = ' '.join(tokens)
        
    return cleaned_text

def main():
    st.set_page_config(layout = "wide", initial_sidebar_state='expanded')
    st.title('Predict whether its spam or not')
    user_input = st.text_input('Enter your text here : ')
    if user_input:
        cleaned_input = cleaning(user_input)
        new_text = loaded_tfidf.transform([cleaned_input])
        prediction = loaded.predict(new_text)
        result = "spam" if prediction[0] == 1 else "not spam"
        st.write(f"This is {result} text")
if __name__ == '__main__':
    main()
