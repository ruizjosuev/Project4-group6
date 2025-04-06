from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import modelHelper

os.chdir(os.path.dirname(os.path.realpath(__file__)))
app = Flask(__name__)

# Load the dataset
df = pd.read_csv("dataset.csv")  # Ensure 'track_name' and 'artists' columns exist

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', danceability=None, error=None)

@app.route('/check', methods=['POST'])
def check_danceability():
    track_name = request.form.get('track_name', '').strip().lower()
    artists = request.form.get('artists', '').strip().lower()

    if not track_name or not artists:
        return render_template('index.html',
                               error="Please enter both track name and artist.",
                               danceability=None,
                               track_name=track_name,
                               artists=artists)

    match = df[
        (df['track_name'].str.lower().str.strip() == track_name) &
        (df['artists'].str.lower().str.strip() == artists)
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
                               artists=artists)

    return render_template('index.html',
                           error="Song not found in the dataset. Try predicting it instead.",
                           danceability=None,
                           track_name=track_name,
                           artists=artists)

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

    # Predict using modelHelper
    prediction = modelHelper.make_predictions(
        energy, loudness, speechiness, acousticness,
        instrumentalness, liveness, valence, tempo,
        key, mode, time_signature
    )

    return jsonify({"ok": True, "prediction": prediction})

if __name__ == '__main__':
    app.run(debug=True)









