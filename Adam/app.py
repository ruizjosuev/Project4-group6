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


@app.route('/top-tracks')
def top_tracks():
    return render_template('top_tracks.html')


@app.route('/visuals')
def visuals():
    return render_template('visuals.html')


@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    danceability_options = [round(x * 0.1, 1) for x in range(11)]
    energy_options = [round(x * 0.1, 1) for x in range(11)]
    recommendations = []

    if request.method == 'POST':
        try:
            # Get user inputs
            danceability = float(request.form.get('danceability'))
            energy = float(request.form.get('energy'))
            tempo = float(request.form.get('tempo'))
            instrumentalness = float(request.form.get('instrumentalness'))

            # Create input vector
            input_vector = [[danceability, energy, tempo, instrumentalness]]

            # Prepare data for similarity
            features = ['danceability', 'energy', 'tempo', 'instrumentalness']
            scaler = StandardScaler()
            X = scaler.fit_transform(df[features])
            input_scaled = scaler.transform(input_vector)

            # Compute cosine similarity
            sim_scores = cosine_similarity(input_scaled, X)[0]
            top_indices = sim_scores.argsort()[-5:][::-1]  # Top 5 recommendations

            recommendations = df.iloc[top_indices][['track_name', 'artists']].to_dict(orient='records')

        except Exception as e:
            print("Error:", e)

    return render_template(
        'custom_recommend.html',
        danceability_options=danceability_options,
        energy_options=energy_options,
        recommendations=recommendations
    )


@app.route('/contact')
def contact():
    return render_template('contact.html')


# Run the app
if __name__ == '__main__':
    app.run(debug=True)


















