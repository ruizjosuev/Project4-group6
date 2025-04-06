import os
import pandas as pd
import pickle


PIPELINE_PATH = os.path.join('..', 'Misha', 'danceability_model_pipeline.h5')


os.chdir(os.path.dirname(os.path.realpath(__file__)))
class ModelHelper:
    def __init__(self):
        with open(PIPELINE_PATH, "rb") as f:
            self.model = pickle.load(f)

        self.class_labels = ["low", "medium", "high"]

    def make_predictions(self, energy, loudness, speechiness, acousticness, instrumentalness,
                         liveness, valence, tempo, key, mode, time_signature):
        df = pd.DataFrame({
            "energy": [energy],
            "loudness": [loudness],
            "speechiness": [speechiness],
            "acousticness": [acousticness],
            "instrumentalness": [instrumentalness],
            "liveness": [liveness],
            "valence": [valence],
            "tempo": [tempo],
            "key": [key],
            "mode": [mode],
            "time_signature": [time_signature]
        })

        preds = self.model.predict(df)
        pred_idx = preds[0]

        if 0 <= pred_idx < len(self.class_labels):
            return self.class_labels[pred_idx]
        return "unknown"

# Instantiate model for reuse
modelHelper = ModelHelper()

