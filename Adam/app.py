from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import modelHelper

os.chdir(os.path.dirname(os.path.realpath(__file__)))
app = Flask(__name__)

# Load the dataset
df = pd.read_csv("cleaned_dataset.csv")  # Make sure this file exists and has 'track_name' & 'artists'

@app.route('/', methods=['GET'])
def index():
    artist_track_map = df.groupby('artists')['track_name'].unique().apply(list).to_dict()
    artist_names = sorted(artist_track_map.keys())

    return render_template('index.html',
                           danceability=None,
                           error=None,
                           artist_names=artist_names,
                           artist_track_map=artist_track_map)

@app.route('/check', methods=['POST'])
def check_danceability():
    track_name = request.form.get('track_name', '').strip()
    artists = request.form.get('artists', '').strip()

    artist_track_map = df.groupby('artists')['track_name'].unique().apply(list).to_dict()
    artist_names = sorted(artist_track_map.keys())

    if not track_name or not artists:
        return render_template('index.html',
                               error="Please enter both track name and artist.",
                               danceability=None,
                               track_name=track_name,
                               artists=artists,
                               artist_names=artist_names,
                               artist_track_map=artist_track_map)

    match = df[
        (df['track_name'].str.strip() == track_name) &
        (df['artists'].str.strip() == artists)
    ]

    if not match.empty:
        song = match.iloc[0]
        return render_template('index.html',
                               danceability=round(song['danceability'], 2),
                               energy=round(song['energy'], 2),
                               tempo=round(song['tempo'], 2),
                               liveness=round(song['liveness'], 2),
                               error=None,
                               track_name=track_name,
                               artists=artists,
                               artist_names=artist_names,
                               artist_track_map=artist_track_map)

    return render_template('index.html',
                           error="Song not found in the dataset. Try another one.",
                           danceability=None,
                           track_name=track_name,
                           artists=artists,
                           artist_names=artist_names,
                           artist_track_map=artist_track_map)

@app.route('/get_tracks', methods=['POST'])
def get_tracks():
    artist = request.json.get('artist', '').strip()
    tracks = df[df['artists'].str.strip() == artist]['track_name'].dropna().unique().tolist()
    return jsonify(tracks=sorted(tracks))

@app.route("/makePredictions", methods=["POST"])
def make_predictions():
    content = request.json["data"]

    # Parse input values
    energy = float(content["energy"])
    loudness = float(content["loudness"])
    speechiness = float(content["speechiness"])
    acousticness = float(content["acousticness"])
    instrumentalness = float(content["instrumentalness"])
    liveness = float(content["liveness"])
    valence = float(content["valence"])
    tempo = float(content["tempo"])
    key = int(content["key"])
    mode = int(content["mode"])
    time_signature = int(content["time_signature"])

    prediction = modelHelper.make_predictions(
        energy, loudness, speechiness, acousticness,
        instrumentalness, liveness, valence, tempo,
        key, mode, time_signature
    )

    return jsonify({"ok": True, "prediction": prediction})

if __name__ == '__main__':
    app.run(debug=True)











