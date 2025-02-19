import nltk
from flask import Flask, jsonify, request
from sklearn.metrics.pairwise import cosine_similarity

from main import calculate_similarity_tfidf, load_tfidf, preprocess


# Initialize Flask app
app = Flask(__name__)

# Download stopwords if not already downloaded
nltk.download("stopwords")


# API endpoint for similarity calculation
@app.route("/calculate_similarity", methods=["POST"])
def calculate_similarity():
    """
    API endpoint for calculating the similarity between two pieces of text.

    The method takes a JSON object with two keys, "text1" and "text2", each
    containing the text to be compared.

    The method will return a JSON response with a single key, "similarity score",
    containing a float value representing the cosine similarity between the two
    texts.

    If the input is invalid, the method will return a JSON response with an "error"
    key containing a string describing the error.

    :param data: A JSON object containing the two texts to be compared
    :return: A JSON response with the similarity score
    :rtype: dict
    """
    try:
        # Get the JSON data from the request
        data = request.get_json()
        # Check if the data is valid
        if not data or "text1" not in data or "text2" not in data:
            return jsonify({"error": "Invalid input format"}), 400
        # Calculate the similarity
        result = calculate_similarity_tfidf(data)
        # Return the result
        return jsonify(result), 200

    except Exception as e:
        # Return an error if something goes wrong
        return jsonify({"error": str(e)}), 400


# Main block to run the app
if __name__ == "__main__":
    app.run(debug=True)
