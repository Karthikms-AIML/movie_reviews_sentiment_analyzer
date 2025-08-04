from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re

app = Flask(__name__)
CORS(app)  # 👈 Add this

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    return text

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    review = clean_text(data["review"])
    vect_review = vectorizer.transform([review])
    prediction = model.predict(vect_review)[0]
    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
    return jsonify({"sentiment": sentiment})

if __name__ == "__main__":
    app.run(debug=True)
