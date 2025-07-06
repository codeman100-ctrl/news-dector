from flask import Flask, render_template, request
import joblib
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import csv
import os
import nltk

nltk.download('punkt_tab')






# Load the trained model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
# Initialize Flask app
app = Flask(__name__)

# Prediction function
def predict_news(text):
    text_vec = vectorizer.transform([text])
    proba = model.predict_proba(text_vec)[0]
    prediction = model.predict(text_vec)[0]

    confidence = round(max(proba) * 100, 2)  # Convert to percentage
    label = "Real News" if prediction == 1 else "Fake News"

    return label, confidence

def summarize_text(text, sentence_count=2):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join([str(sentence) for sentence in summary])


# Routes
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None
    summary = None
    feedback = None

    if request.method == "POST":
        user_input = request.form.get("news")
        feedback = request.form.get("feedback")

        if feedback:  # Save feedback
            with open("feedback_log.csv", mode="a", newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([user_input, feedback])
            return render_template("index.html", feedback_thank_you=True)

        result, confidence = predict_news(user_input)
        summary = summarize_text(user_input)

        return render_template("index.html",
                               prediction=result,
                               confidence=confidence,
                               summary=summary,
                               user_input=user_input)

    return render_template("index.html")




if __name__ == "__main__":
    app.run(debug=True)
