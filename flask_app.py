import nltk
from flask import Flask, jsonify, request
from sklearn.metrics.pairwise import cosine_similarity

from main import calculate_similarity_tfidf, load_tfidf, preprocess

# Load the model
# m = "sentence-transformers/all-MiniLM-L6-v2"
# embedding = HuggingFaceEmbeddings(model_name=m)

# Initialize Flask app
app = Flask(__name__)

# Download stopwords if not already downloaded
nltk.download("stopwords")


# API endpoint for similarity calculation
@app.route("/calculate_similarity", methods=["POST"])
def calculate_similarity():
    try:
        data = request.get_json()
        if not data or "text1" not in data or "text2" not in data:
            return jsonify({"error": "Invalid input format"}), 400
        result = calculate_similarity_tfidf(data)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Main block to run the app
if __name__ == "__main__":
    app.run(debug=True)
