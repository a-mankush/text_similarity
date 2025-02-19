from flask import Flask, jsonify, request
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# Load the model
m = "sentence-transformers/all-MiniLM-L6-v2"
embedding = HuggingFaceEmbeddings(model_name=m)

# Initialize Flask app
app = Flask(__name__)


# Similarity calculation function
def cal_sim(data: dict):
    vec1 = embedding.embed_query(data["text1"])
    vec2 = embedding.embed_query(data["text2"])
    similarity = cosine_similarity([vec1], [vec2])[0][0]
    return {"similarity": float(similarity)}


# API endpoint for similarity calculation
@app.route("/calculate_similarity", methods=["POST"])
def calculate_similarity():
    try:
        data = request.get_json()
        if not data or "text1" not in data or "text2" not in data:
            return jsonify({"error": "Invalid input format"}), 400
        result = cal_sim(data)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Main block to run the app
if __name__ == "__main__":
    app.run(debug=True)
