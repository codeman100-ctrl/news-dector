from flask import Flask, render_template, request , redirect
import joblib
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import csv
import os
import nltk
import requests
from urllib.parse import quote
from sentence_transformers import SentenceTransformer, util
import re

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address





nltk.download('punkt_tab')
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')



# api_key = "52395907dd394464b923a73e76ffc2d9"



# Load the trained model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
# Initialize Flask app
app = Flask(__name__)

# IP-based limiter setup
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[]
)

# Prediction function
def predict_news(text):
    text_vec = vectorizer.transform([text])
    proba = model.predict_proba(text_vec)[0]
    prediction = model.predict(text_vec)[0]

    confidence = round(max(proba) * 100, 2)
    label = "Real News" if prediction == 1 else "Fake News"

    return label, confidence

def summarize_text(text, sentence_count=2):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join([str(sentence) for sentence in summary])

def get_trusted_sources(user_text, api_key):
    # Step 1: Clean & shorten input
    query = ' '.join(user_text.split()[:10])
    encoded_query = quote(query)
    
    url = f"https://newsapi.org/v2/everything?q={encoded_query}&language=en&sortBy=relevancy&pageSize=5&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code != 200:
        return []

    data = response.json()
    user_inputs = data.get("user_inputs", [])
    if not user_inputs:
        return []

    # Step 2: Semantic comparison
    user_embedding = semantic_model.encode(user_text, convert_to_tensor=True)
    results = []

    for user_input in user_inputs:
        title = user_input["title"]
        source = user_input["source"]["name"]
        url = user_input["url"]

        title_embedding = semantic_model.encode(title, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(user_embedding, title_embedding).item()

        if similarity > 0.5:  # Threshold for relevance
            results.append({
                "title": title,
                "source": source,
                "url": url,
                "similarity": round(similarity * 100, 2)
            })

    # Return sorted by similarity
    return sorted(results, key=lambda x: x["similarity"], reverse=True)

def get_explanation(text, trusted_sources):
    explanations = []

    # 1. Suspicious / clickbait keywords
    emotional_keywords = ['shocking', 'miracle', 'cure', 'exposed', 'urgent', 'breaking', 'secret', 'you won’t believe', 'alert']
    found_keywords = [word for word in emotional_keywords if word in text.lower()]
    if found_keywords:
        explanations.append(f"Contains emotional/clickbait terms: {', '.join(found_keywords)}")

    # 2. ALL CAPS detection
    if re.search(r'\b[A-Z]{4,}\b', text):
        explanations.append("Contains ALL CAPS words, often seen in misleading news")

    # 3. Trusted sources match
    if not trusted_sources:
        explanations.append("No matching trusted sources found")

    # 4. Exaggerated punctuation
    if "!!" in text or "??" in text:
        explanations.append("Excessive punctuation used")

    # 5. Short length (can indicate poor credibility)
    if len(text.split()) < 30:
        explanations.append("Text is very short — may lack credible detail")

    # If none of the above
    if not explanations:
        explanations.append("No strong suspicious patterns found in the text")

    return explanations

def save_feedback(user_input, prediction, confidence, was_correct):
    with open("feedback_data.csv", mode="a", newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            user_input.strip(),
            prediction,
            round(confidence, 4),
            was_correct,
            datetime.now().isoformat()
        ])

# Function to get Google News results from SerpAPI
def get_serpapi_results(query, serpapi_key, top_k=3):
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "api_key": serpapi_key,
        "engine": "google",
        "tbm": "nws",  # News tab
        "num": 10
    }

    try:
        response = requests.get(url, params=params)
        results = response.json().get("news_results", [])

        # Semantic similarity matching
        user_embedding = semantic_model.encode(query, convert_to_tensor=True)
        final_results = []

        for article in results:
            title = article.get("title", "")
            snippet = article.get("snippet", "")
            link = article.get("link", "")
            source = article.get("source", "")
            combined_text = f"{title}. {snippet}"
            similarity = util.cos_sim(user_embedding, semantic_model.encode(combined_text, convert_to_tensor=True)).item()

            final_results.append({
                "title": title,
                "snippet": snippet,
                "link": link,
                "source": source,
                "similarity": round(similarity * 100, 2)
            })

            icon = get_source_icon(source)
            final_results.append({
                "title": title,
                "snippet": snippet[:200] + "..." if len(snippet) > 200 else snippet,
                "link": link,
                "source": f"{icon} {source}",
                "similarity": round(similarity * 100, 2)
            })


        # Return top-k sorted by similarity
        return sorted(final_results, key=lambda x: x['similarity'], reverse=True)[:top_k]

    except Exception as e:
        print(f"Error in SerpAPI search: {e}")
        return []
def get_source_icon(source_name):
    icons = {
        "BBC": "📰",
        "CNN": "🌍",
        "NDTV": "🧭",
        "India Today": "🇮🇳",
        "The Hindu": "📜",
        "Times of India": "🕰️",
        "Al Jazeera": "🕌",
        "Reuters": "🧾",
        "The Guardian": "🛡️"
    }
    for key in icons:
        if key.lower() in source_name.lower():
            return icons[key]
    return "🔎"  # default icon

# Routes
@app.route("/", methods=["GET", "POST"])
@limiter.limit("5 per minute", error_message="⛔ Too many prediction requests. Please slow down and try again shortly.")
def home():
    result = None
    confidence = None
    summary = None
    feedback = None
    trusted_sources = None

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
        # trusted_sources = get_trusted_sources(user_input, "52395907dd394464b923a73e76ffc2d9")
        # explanation = get_explanation(user_input, trusted_sources)
        # Inside your route
        label, confidence = predict_news(user_input)
        trusted_sources = get_trusted_sources(user_input, "52395907dd394464b923a73e76ffc2d9")
        explanation = get_explanation(user_input, trusted_sources)
        trusted_sources = get_serpapi_results(user_input, serpapi_key="56b1cd4d0e1d17dad526da85b650ccd7fa6eb8caeccc9e0ee89232efb4af65b0")





        # ⛔ Lightweight Override Logic Here
        for source in trusted_sources:
            if source['similarity'] > 55:
                label = "Real News"
                confidence = source['similarity']
                explanation.insert(0, f"✅ Trusted source '{source['source']}' matched with {source['similarity']}% similarity — overriding model prediction.")
                break   



        return render_template("index.html",
                               prediction=result,
                               confidence=confidence,
                               summary=summary,
                               user_input=user_input,
                               trusted_sources=trusted_sources,
                               
                               feedback_thank_you=False)

    return render_template("index.html")



@app.route("/submit_feedback", methods=["POST"])
@limiter.limit("5 per minute", error_message="⛔ Too many prediction requests. Please slow down and try again shortly.")
def submit_feedback():
    user_input = request.form.get("user_input")
    prediction = request.form.get("prediction")
    confidence = request.form.get("confidence")
    was_correct = request.form.get("feedback") == "yes"

    if not all([user_input, prediction, confidence]):
        return "Invalid form submission", 400

    save_feedback(user_input, prediction, float(confidence), was_correct)

    return redirect("/")  # Or return to index.html with a thank-you flag

@app.errorhandler(429)
def ratelimit_handler(e):
    return render_template("index.html", error=str(e.description)), 429




if __name__ == "__main__":
    app.run(debug=True)
