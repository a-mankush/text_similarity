from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity


def get_model():
    print("Loading model...")
    m = "sentence-transformers/all-MiniLM-L6-v2"
    embedding = HuggingFaceEmbeddings(model_name=m)
    print("Model loaded")
    return embedding


# Get the cached model


def cal_sim(data: dict, embedding_moddel: HuggingFaceEmbeddings):
    vec1 = embedding_moddel.embed_query(data["text1"])
    vec2 = embedding_moddel.embed_query(data["text2"])
    return {"similarity": float(cosine_similarity([vec1], [vec2])[0][0])}


if __name__ == "__main__":

    embedding = get_model()
    data = {
        "text1": "Ram Runs on the train every morning",
        "text2": "Ram have strong legs because he runs every day",
    }

    result = cal_sim(data)

    print(result)
