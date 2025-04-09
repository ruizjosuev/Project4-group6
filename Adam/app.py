from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Setup
os.chdir(os.path.dirname(os.path.realpath(__file__)))
app = Flask(__name__)

# Load dataset
df = pd.read_csv("cleaned_dataset.csv")  # Ensure required columns exist

# ---------- ROUTES ----------

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/danceability', methods=['GET', 'POST'])
def danceability():
    artist_track_map = df.groupby('artists')['track_name'].unique().apply(list).to_dict()
    artist_names = sorted(artist_track_map.keys())
    danceability = error = theme_class = None
    track_name = artists = ""

    if request.method == 'POST':
        track_name = request.form.get('track_name', '').strip()
        artists = request.form.get('artists', '').strip()

        if not track_name or not artists:
            error = "Please select both artist and track."
        else:
            match = df[
                (df['track_name'].str.strip() == track_name) &
                (df['artists'].str.strip() == artists)
            ]

            if not match.empty:
                song = match.iloc[0]
                danceability = song['danceability']

                if danceability < 0.4:
                    theme_class = 'low-theme'
                elif danceability < 0.7:
                    theme_class = 'medium-theme'
                else:
                    theme_class = 'high-theme'

                return render_template(
                    'danceability.html',
                    danceability=round(danceability, 2),
                    energy=round(song['energy'], 2),
                    tempo=round(song['tempo'], 2),
                    liveness=round(song['liveness'], 2),
                    theme_class=theme_class,
                    error=None,
                    track_name=track_name,
                    artists=artists,
                    artist_names=artist_names,
                    artist_track_map=artist_track_map
                )
            else:
                error = "Song not found in the dataset."

    return render_template(
        'danceability.html',
        danceability=danceability,
        error=error,
        theme_class=theme_class,
        track_name=track_name,
        artists=artists,
        artist_names=artist_names,
        artist_track_map=artist_track_map
    )


@app.route('/get_tracks', methods=['POST'])
def get_tracks():
    artist = request.json.get('artist', '').strip()
    tracks = df[df['artists'].str.strip() == artist]['track_name'].dropna().unique().tolist()
    return jsonify(tracks=sorted(tracks))


@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    danceability_options = [round(x * 0.1, 1) for x in range(11)]
    energy_options = [round(x * 0.1, 1) for x in range(11)]
    recommendations = []

    if request.method == 'POST':
        try:
            danceability = float(request.form.get('danceability'))
            energy = float(request.form.get('energy'))
            tempo = float(request.form.get('tempo'))
            instrumentalness = float(request.form.get('instrumentalness'))

            input_vector = [[danceability, energy, tempo, instrumentalness]]
            features = ['danceability', 'energy', 'tempo', 'instrumentalness']
            scaler = StandardScaler()
            X = scaler.fit_transform(df[features])
            input_scaled = scaler.transform(input_vector)

            sim_scores = cosine_similarity(input_scaled, X)[0]
            top_indices = sim_scores.argsort()[-5:][::-1]
            recommendations = df.iloc[top_indices][['track_name', 'artists']].to_dict(orient='records')

        except Exception as e:
            print("Error:", e)

    return render_template(
        'custom_recommend.html',
        danceability_options=danceability_options,
        energy_options=energy_options,
        recommendations=recommendations
    )


@app.route('/top-tracks')
def top_tracks():
    return render_template('top_tracks.html')


@app.route('/visuals')
def visuals():
    return render_template('visuals.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


# ---------- Run App ----------
if __name__ == '__main__':
    app.run(debug=True)



















