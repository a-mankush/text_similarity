import pickle
import string

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def train_and_dump_vect():

    # Read the dataset from a CSV file
    df = pd.read_csv("DataNeuron_Text_Similarity.csv")

    # Split the dataframe into two numpy arrays, one for text1 and one for text2
    text1 = df["text1"].to_numpy()
    text2 = df["text2"].to_numpy()

    # Concatenate the two arrays into a single list
    data_list = list(text1) + list(text2)

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Preprocess the data using the preprocess() function
    preprocessed_data = [preprocess(text) for text in data_list]

    # Fit the vectorizer to the preprocessed data
    vectorizer.fit(preprocessed_data)

    # Store the vectorizer in a pickle file
    pickle.dump(vectorizer, open("vectorizer.pickle", "wb"))


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
        "text1": "nuclear body seeks new tech",
        "text2": "terror suspects face arrest",
    }

    similarity_score = calculate_similarity_tfidf(data)
    print(f"Similarity score: {similarity_score:.4f}")
