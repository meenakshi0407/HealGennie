import nltk
import streamlit as st
import random
import os
import json
import csv
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt', quiet=True)

file_path = os.path.abspath("./intents.json")
with open(file_path, encoding='utf-8') as file:
    intents = json.load(file)

vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=1000)

tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

CRISIS_KEYWORDS = [
    "suicide", "kill myself", "self-harm", "end my life",
    "hurt myself", "want to die", "no reason to live"
]

CRISIS_RESPONSE = (
    "💙 I'm really sorry you're feeling this way. You're not alone. "
    "Please reach out to **iCall** at **9152987821** (India) or visit "
    "[icallhelpline.org](https://icallhelpline.org) for immediate support. "
    "If you're in immediate danger, please call **112**."
)

def chatbot(input_text):
    input_lower = input_text.lower()

    if any(keyword in input_lower for keyword in CRISIS_KEYWORDS):
        return CRISIS_RESPONSE

    input_vec = vectorizer.transform([input_lower])
    proba = clf.predict_proba(input_vec).max()
    tag = clf.predict(input_vec)[0]

    # st.write(f"DEBUG — Tag: {tag} | Confidence: {proba:.2f}")

    if proba < 0.1:
        return "🤔 I'm not quite sure I understand. Could you rephrase that or ask something else about health and wellness?"

    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

    return "I'm not sure how to respond to that. Try asking something related to health, fitness, or wellness!"

CSV_FILE = "chat_log.csv"

def init_csv():
    """Create CSV with header if it doesn't exist."""
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline='', encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

def log_conversation(user_input, response):
    """Append a conversation turn to the CSV log."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([user_input, response, timestamp])

def main():
    st.set_page_config(page_title="HealGennie", page_icon="🌿", layout="centered")
    st.title("🌿 HealGennie")
    st.caption("Your personal Health & Wellness chatbot — ask me anything about health, fitness, and wellbeing!")

    init_csv()

    menu = ["💬 Chat", "📜 Conversation History", "ℹ️ About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "💬 Chat":

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for role, msg in st.session_state.messages:
            with st.chat_message("user" if role == "You" else "assistant"):
                st.markdown(msg)

        user_input = st.chat_input("Type your health question here...")

        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)

            response = chatbot(user_input)
            with st.chat_message("assistant"):
                st.markdown(response)

            st.session_state.messages.append(("You", user_input))
            st.session_state.messages.append(("HealGennie", response))

          
            log_conversation(str(user_input), response)


            if any(word in response.lower() for word in ['bye', 'goodbye']):
                st.success("Thank you for chatting with me. Have a great day! 💚")
                st.stop()

    elif choice == "📜 Conversation History":
        st.header("📜 Conversation History")

        if not os.path.exists(CSV_FILE):
            st.info("No conversation history yet. Start chatting first!")
        else:
            with open(CSV_FILE, 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip header row
                rows = list(csv_reader)

            if not rows:
                st.info("No conversations recorded yet.")
            else:
                for row in reversed(rows):  # Most recent first
                    with st.expander(f"🕐 {row[2]}"):
                        st.markdown(f"**You:** {row[0]}")
                        st.markdown(f"**HealGennie:** {row[1]}")

    elif choice == "ℹ️ About":
        st.header("ℹ️ About HealGennie")
        st.write(
            "HealGennie is a chatbot designed to understand and respond to user queries "
            "related to health and wellness. It uses **Natural Language Processing (NLP)** "
            "and **Logistic Regression** to classify user intent and provide helpful responses."
        )

        st.subheader("🔧 How It Works")
        st.write("""
        1. **NLP + TF-IDF**: User input is vectorized using TF-IDF with n-grams (1–4).
        2. **Logistic Regression**: The classifier predicts the intent from the vectorized input.
        3. **Confidence Threshold**: If confidence < 40%, the bot asks for clarification.
        4. **Safety Filter**: Crisis-related inputs are redirected to professional helplines.
        5. **Streamlit UI**: A modern chat interface with full conversation history.
        """)

        st.subheader("📦 Tech Stack")
        st.markdown("""
        - `Python` — Core language
        - `scikit-learn` — TF-IDF Vectorizer + Logistic Regression
        - `NLTK` — Natural Language Toolkit
        - `Streamlit` — Web interface
        - `CSV` — Conversation logging
        """)

        st.subheader("🚀 Future Improvements")
        st.write("""
        - Add deep learning models (e.g., BERT, sentence-transformers) for better accuracy
        - Support Hindi and regional Indian languages
        - Add a symptom checker flow
        - Integrate verified health resource links
        """)

if __name__ == '__main__':
    main()