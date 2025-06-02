import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    # Remove non_alphanumeric character
    y= [i for i in text if i.isalnum()]

    # Remove stopwords likw "a","am", etc and punctuation
    stop_words = set(stopwords.words("english"))
    text = [i for i in text if i not in stop_words and i not in string.punctuation]

    # stemming
    text = [ps.stem(i) for i in text]
    return " ".join(text)

tfidv = pickle.load(open('vectorizer.pk1', 'rb'))
model = pickle.load(open("model.pkl",'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_input("Enter the message")

if st.button("Predict"):
    transform_sms = transform_text(input_sms)
    vector_input = tfidv.transform([transform_sms])
    result = model.predict(vector_input)[0]
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")