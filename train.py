import pandas as pd
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# LOAD DATASET
# ==========================================

songs = pd.read_csv("songs.csv")

# ==========================================
# KEEP IMPORTANT COLUMNS
# ==========================================

songs = songs[
    [
        "track_name",
        "artists",
        "track_genre"
    ]
]

# ==========================================
# REMOVE NULL VALUES
# ==========================================

songs.dropna(inplace=True)

# ==========================================
# USE SMALL DATASET
# ==========================================

songs = songs.head(500)

# ==========================================
# REMOVE DUPLICATES
# ==========================================

songs.drop_duplicates(
    subset="track_name",
    inplace=True
)

# ==========================================
# CREATE TAGS COLUMN
# ==========================================

songs["tags"] = (
    songs["artists"].astype(str) + " " +
    songs["track_genre"].astype(str)
)

# ==========================================
# CONVERT TO LOWERCASE
# ==========================================

songs["tags"] = songs["tags"].apply(
    lambda x: x.lower()
)

# ==========================================
# TEXT VECTORIZATION
# ==========================================

cv = CountVectorizer(
    max_features=2000,
    stop_words="english"
)

vectors = cv.fit_transform(
    songs["tags"]
).toarray()

# ==========================================
# COSINE SIMILARITY
# ==========================================

similarity = cosine_similarity(vectors)

# ==========================================
# SAVE FILES
# ==========================================

pickle.dump(
    songs,
    open("songs.pkl", "wb")
)

pickle.dump(
    similarity,
    open("similarity.pkl", "wb")
)

print("Model Trained Successfully")
