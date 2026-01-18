"""
Problem with this approach:
1. compute distance of the query vector with all possible docs is very time consuming
2. Also we need to store each of these vectors in their raw form leading to high memory consumption.
"""

from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


docs = [
    "Pad Thai is a popular Thai stir-fried noodle dish with tamarind sauce, peanuts, and lime",
    "Sushi is a Japanese dish featuring vinegared rice combined with raw fish and vegetables",
    "Biryani is a fragrant Indian rice dish cooked with aromatic spices, meat, and saffron",
    "Pho is a Vietnamese soup with rice noodles, herbs, and beef or chicken broth",
    "Ramen consists of Chinese-style wheat noodles served in a flavorful meat or fish broth",
    "Dim sum includes various Chinese small bite-sized portions served in steamer baskets",
    "Kimchi is a traditional Korean fermented vegetable dish made primarily with napa cabbage",
    "Tom Yum is a hot and sour Thai soup with lemongrass, lime leaves, and shrimp",
    "Dumplings are filled dough pockets found across Asian cuisines with meat or vegetable fillings",
    "Butter chicken is a creamy Indian curry dish with tender chicken in tomato-based sauce"
]


def get_embedding(text, model="text-embedding-3-small"):
    embedding = client.embeddings.create(input=[text], model=model).data[0].embedding
    return embedding

def calc_cosine_similarity(query_vec, vec):
    return np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))


sentence_vectors = {}
for doc in docs:
    embedding = get_embedding(doc)
    sentence_vectors[doc] = embedding

def get_top_n_similar(query_sentence, n=2):
    query_vec = get_embedding(text=query_sentence)

    similarities = {sentence: calc_cosine_similarity(query_vec=query_vec, vec=sentence_vectors[sentence]) for sentence in docs}

    sorted_similarities = dict(sorted(similarities.items(), key=lambda x: x[1], reverse=True))
    top_matches = list(sorted_similarities.items())[:n]

    for sentence, score in top_matches:
        print(f"Similarity: {score:.4f} - {sentence}")

if __name__ == "__main__":
    # query_sentence = "I wanna eat some soup with noodles."
    query_sentence = "I am craving for some protein. What kind of food items I can eat."
    get_top_n_similar(query_sentence=query_sentence, n=2)
