import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import argparse

def load_dataset(filepath):
    """
    Load the movie dataset from a CSV file.
    """
    df = pd.read_csv(filepath)
    return df[['Title', 'Plot']]

def compute_similarity(user_input, dataset):
    """
    Compute cosine similarity between user input and dataset movie plots.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(dataset['Plot'].tolist() + [user_input])
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    return cosine_sim

def recommend_movies(user_input, dataset, top_n=5):
    """
    Recommend top N movies based on similarity to user input.
    """
    similarity_scores = compute_similarity(user_input, dataset)
    dataset['similarity'] = similarity_scores
    recommendations = dataset.sort_values(by='similarity', ascending=False).head(top_n)
    return recommendations[['Title', 'similarity']]

def main():
    parser = argparse.ArgumentParser(description='Content-Based Movie Recommendation System')
    parser.add_argument('query', type=str, help='User input describing movie preferences')
    parser.add_argument('dataset_path', type=str, help='Path to the movie dataset CSV file')
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_path)
    recommendations = recommend_movies(args.query, dataset)

    print("Recommended Movies:")
    print(recommendations)

if __name__ == "__main__":
    main()