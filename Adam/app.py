from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import modelHelper

# Setup
os.chdir(os.path.dirname(os.path.realpath(__file__)))
app = Flask(__name__)

# Load dataset
df = pd.read_csv("cleaned_dataset.csv")  # Ensure 'track_name' and 'artists' columns exist

# ---------- ROUTES ----------

@app.route('/')
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
                               error="Please select both artist and track.",
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
        danceability = song['danceability']

        # Determine result theme based on danceability
        if danceability < 0.4:
            theme_class = 'low-theme'
        elif danceability < 0.7:
            theme_class = 'medium-theme'
        else:
            theme_class = 'high-theme'

        return render_template('index.html',
                               danceability=round(danceability, 2),
                               energy=round(song['energy'], 2),
                               tempo=round(song['tempo'], 2),
                               liveness=round(song['liveness'], 2),
                               theme_class=theme_class,
                               error=None,
                               track_name=track_name,
                               artists=artists,
                               artist_names=artist_names,
                               artist_track_map=artist_track_map)

    return render_template('index.html',
                           error="Song not found in the dataset.",
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


@app.route('/makePredictions', methods=['POST'])
def make_predictions():
    content = request.json["data"]

    prediction = modelHelper.make_predictions(
        float(content["energy"]),
        float(content["loudness"]),
        float(content["speechiness"]),
        float(content["acousticness"]),
        float(content["instrumentalness"]),
        float(content["liveness"]),
        float(content["valence"]),
        float(content["tempo"]),
        int(content["key"]),
        int(content["mode"]),
        int(content["time_signature"])
    )

    return jsonify({"ok": True, "prediction": prediction})


@app.route('/contact')
def contact():
    return render_template('contact.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)













