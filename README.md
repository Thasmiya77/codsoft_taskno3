# Task 3 - Recommendation System
### CodSoft Artificial Intelligence Internship

A **content-based movie recommendation system** that suggests similar
movies based on genres and description text - no user ratings needed.

## How it works
1. Each movie's genres + description are combined into a single text
   "profile".
2. All profiles are converted into **TF-IDF vectors** using
   `scikit-learn`'s `TfidfVectorizer` - this turns text into numbers,
   giving more weight to distinctive words and less to common ones.
3. **Cosine similarity** is computed between every pair of movies,
   producing a score between 0 (unrelated) and 1 (identical content).
4. Given a movie the user likes, the system returns the top N movies
   with the highest similarity score.

This is **content-based filtering** (as opposed to collaborative
filtering, which relies on other users' behavior/ratings).

## Files
- `recommender.py` - main script
- `movies.csv` - small sample dataset (20 movies with genres + description)

## Run it
```bash
pip install pandas scikit-learn
python3 recommender.py
```

## Example
```
Enter a movie you like: The Matrix

Because you liked 'The Matrix', you might also enjoy:
  - Guardians of the Galaxy  (Action Sci-Fi Comedy Adventure)  [similarity: 0.159]
  - Avengers Endgame  (Action Sci-Fi Adventure)  [similarity: 0.144]
  - Inception  (Action Sci-Fi Thriller)  [similarity: 0.131]
  - Interstellar  (Sci-Fi Drama)  [similarity: 0.106]
  - John Wick  (Action Thriller)  [similarity: 0.048]
```

## Extending it
- Swap `movies.csv` for a larger real-world dataset (e.g. MovieLens or
  TMDB) for richer recommendations.
- Add weighting so genre matches count more than description matches.
- Combine with collaborative filtering for a hybrid recommender.

## Concepts demonstrated
- Content-based filtering
- TF-IDF text vectorization
- Cosine similarity
- `pandas` for data handling, `scikit-learn` for ML utilities

---
Built as part of the CodSoft AI Internship (Task 4).
