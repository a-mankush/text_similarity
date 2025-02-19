import pickle
import string

import nltk

# from langchain_huggingface import HuggingFaceEmbeddings
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# def get_model():
#     print("Loading model...")
#     m = "sentence-transformers/all-MiniLM-L6-v2"
#     embedding = HuggingFaceEmbeddings(model_name=m)
#     print("Model loaded")
#     return embedding


# # Similarity calculation function
# def cal_sim_with_hf(data: dict):
#     vec1 = embedding.embed_query(data["text1"])
#     vec2 = embedding.embed_query(data["text2"])
#     similarity = cosine_similarity([vec1], [vec2])[0][0]
#     return {"similarity": float(similarity)}


def preprocess(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]
    return " ".join(words)


def load_tfidf(path: str):
    return pickle.load(open(path, "rb"))


def calculate_similarity_tfidf(data: dict):
    # Preprocess both texts
    preprocessed1 = preprocess(data["text1"])
    preprocessed2 = preprocess(data["text2"])

    # load TF-IDF vectors
    loaded_vect = load_tfidf("vectorizer.pickle")
    tfidf_matrix = loaded_vect.transform([preprocessed1, preprocessed2])

    # Calculate cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return {"similarity score": similarity[0][0]}


if __name__ == "__main__":

    # embedding = get_model()
    data = {
        "text1": "Ram Runs on the train every morning",
        "text2": "Ram have strong legs because he runs every day",
    }
    # result = cal_sim_with_hf(data)
    # print(result)

    similarity_score = calculate_similarity_tfidf(data)
    print(f"Similarity score: {similarity_score:.4f}")
