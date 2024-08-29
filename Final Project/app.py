import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your pre-trained model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vect = pickle.load(vectorizer_file)

# Define the Streamlit app
st.title('Tweet Sentiment Classifier')

st.write('Enter a tweet to classify its sentiment:')

# Input text box for the user to enter a tweet
user_input = st.text_area("Tweet", "")

# Button to make predictions
if st.button('Classify'):
    if user_input:
        # Transform the input tweet using the loaded vectorizer
        user_input_vectorized = vect.transform([user_input])
        
        # Predict sentiment
        prediction = model.predict(user_input_vectorized)
        
        # Display the result
        sentiment = "Positive" if prediction[0] == 4 else "Negative"
        st.write(f'The sentiment of the tweet is: {sentiment}')
    else:
        st.write("Please enter a tweet for classification.")
