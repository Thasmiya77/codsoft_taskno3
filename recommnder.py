"""
CODSOFT - Artificial Intelligence Internship
Task 4: Recommendation System

A content-based movie recommendation system. Instead of relying on
other users' ratings (collaborative filtering), this looks at the
CONTENT of each movie - its genres and description - and recommends
movies that are textually most similar to one the user already likes.

Approach:
1. Combine each movie's genres + description into one text "profile".
2. Convert all movie profiles into TF-IDF vectors (numerical
   representations that weigh distinctive words more heavily than
   common ones).
3. Compute cosine similarity between every pair of movies - a score
   from 0 (completely different) to 1 (identical) based on the angle
   between their TF-IDF vectors.
4. Given a movie the user likes, recommend the N most similar movies.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data(path="movies.csv"):
    df = pd.read_csv(path)
    df["profile"] = df["genres"] + " " + df["description"]
    return df


def build_similarity_matrix(df):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["profile"])
    similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return similarity


def recommend(title, df, similarity, top_n=5):
    matches = df[df["title"].str.lower() == title.lower()]
    if matches.empty:
        return None

    idx = matches.index[0]
    scores = list(enumerate(similarity[idx]))
    # Exclude the movie itself, sort by similarity score descending
    scores = [s for s in scores if s[0] != idx]
    scores.sort(key=lambda x: x[1], reverse=True)

    top_matches = scores[:top_n]
    results = df.iloc[[i for i, _ in top_matches]][["title", "genres"]].copy()
    results["similarity"] = [round(score, 3) for _, score in top_matches]
    return results.reset_index(drop=True)


def list_movies(df):
    print("\nAvailable movies:")
    for title in df["title"]:
        print(f"  - {title}")
    print()


def main():
    df = load_data()
    similarity = build_similarity_matrix(df)

    print("=== Content-Based Movie Recommender ===")
    list_movies(df)

    while True:
        title = input("Enter a movie you like (or 'list' / 'quit'): ").strip()

        if title.lower() == "quit":
            print("Goodbye!")
            break
        if title.lower() == "list":
            list_movies(df)
            continue

        results = recommend(title, df, similarity, top_n=5)
        if results is None:
            print(f"'{title}' not found. Type 'list' to see available movies.\n")
            continue

        print(f"\nBecause you liked '{title}', you might also enjoy:")
        for _, row in results.iterrows():
            print(f"  - {row['title']}  ({row['genres']})  "
                  f"[similarity: {row['similarity']}]")
        print()


if __name__ == "__main__":
    main()
