from flask import Flask, render_template, request
import pickle

# ==========================================
# CREATE FLASK APP
# ==========================================

app = Flask(__name__)

# ==========================================
# LOAD MODEL FILES
# ==========================================

songs = pickle.load(open("songs.pkl", "rb"))
similarity = pickle.load(open("similarity.pkl", "rb"))

# ==========================================
# RECOMMEND FUNCTION (SAFE VERSION)
# ==========================================

def recommend(song_name):

    song_name = song_name.lower()

    # find song index safely
    matching_songs = songs[songs["track_name"].str.lower() == song_name]

    if matching_songs.empty:
        return ["Song not found"]

    song_index = matching_songs.index[0]

    distances = similarity[song_index]

    song_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_songs = []

    for i in song_list:
        recommended_songs.append(
            songs.iloc[i[0]]["track_name"]
        )

    return recommended_songs

# ==========================================
# HOME ROUTE
# ==========================================

@app.route("/", methods=["GET", "POST"])
def home():

    recommendations = []

    if request.method == "POST":
        song = request.form["song"]
        recommendations = recommend(song)

    return render_template(
        "index.html",
        songs=songs["track_name"].values,
        recommendations=recommendations
    )

# ==========================================
# RUN APP
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)
