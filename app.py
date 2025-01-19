import nltk
import streamlit as st
import random
import ssl
import os
import json
import csv
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl.create_default_http_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

#HealGenie
file_path=os.path.abspath("./intents.json")
with open('intents.json', encoding='utf-8') as file:
  intents = json.load(file)

#create vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0,max_iter=1000)

#preprocess the data 
tags=[]
patterns=[]
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)


#train the model
x=vectorizer.fit_transform(patterns)
y=tags
clf.fit(x,y)

from scipy.sparse import csr_matrix

sparse_matrix= csr_matrix(x)
#print(sparse_matrix)

def chatbot(input_text):
    input_text = input_text.lower()
    input_text=vectorizer.transform([input_text])
    tag=clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response= random.choice(intent['responses'])
            return response

counter=0

def main():
    global counter
    st.title("HealGennie")
   
    menu=["Home","Conversation history","About"]
    choice=st.sidebar.selectbox("Menu",menu)

    if choice=='Home':
      st.write("Welcome to the HealGennie chatbot. Please type a message and press Enter to start the conversation.")

      if not os.path.exists("chat_log.csv"):
        with open("chat_log.csv","w",newline='',encoding="utf-8") as csvfile:
          csv_writer=csv.writer(csvfile)
          csv.writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

      counter+=1

      user_input = st.text_input("You:", key=f"user_input_{counter}")
      if user_input:

        user_input_str = str(user_input)

        response=chatbot(user_input)

        st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")
        timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

        with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
          csv_writer = csv.writer(csvfile)
          csv_writer.writerow([user_input_str, response, timestamp])

        if response.lower()==['bye','goodbye']:
            st.write("Thank you for chatting with me. Have a great day!")
            st.stop()

    elif choice=="Conversation history":
      st.header("Conversation History")
       
      with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
          st.text(f"User: {row[0]}")
          st.text(f"Chatbot: {row[1]}")
          st.text(f"Timestamp: {row[2]}")
          st.markdown("---")
    elif choice == "About":
      st.write("The goal of this project is to create a chatbot that can understand and respond to user input based on intents. The chatbot is built using Natural Language Processing (NLP) library and Logistic Regression, to extract the intents and entities from user input. The chatbot is built using Streamlit, a Python library for building interactive web applications.")

      st.subheader("Project Overview:")

      st.write("""
        The project is divided into two parts:
        1. NLP techniques and Logistic Regression algorithm is used to train the chatbot on labeled intents and entities.
        2. For building the Chatbot interface, Streamlit web framework is used to build a web-based chatbot interface. The interface allows users to input text and receive responses from the chatbot.
        """)

      st.subheader("Dataset:")

      st.write("""
        The dataset used in this project is a collection of labelled intents and entities. The data is stored in a list.
        - Intents: The intent of the user input (e.g. "greeting", "How can I lose weight?", "about")
        - Entities: The entities extracted from user input (e.g. "Hi", "How much I drink water?", "I have anxiety?")
        - Text: The user input text.
        """)

      st.subheader("Streamlit Chatbot Interface:")

      st.write("The chatbot interface is built using Streamlit. The interface includes a text input box for users to input their text and a chat window to display the chatbot's responses. The interface uses the trained model to generate responses to user input.")

      st.subheader("Conclusion:")

      st.write("In this project, a chatbot is built that can understand and respond to user input based on intents. The chatbot was trained using NLP and Logistic Regression, and the interface was built using Streamlit. This project can be extended by adding more data, using more sophisticated NLP techniques, deep learning algorithms.")

if __name__ == '__main__':
    main()
       
    