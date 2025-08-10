from flask import Flask, request, jsonify, render_template
import os
import librosa
import numpy as np
import pickle
import traceback
from sklearn.ensemble import RandomForestClassifier
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow frontend JS to access backend responses

UPLOAD_FOLDER = "uploads"
DATASET_FOLDER = "dataset"
MODEL_FILE = "crunchiness_model.pkl"
LABELS = ["soft", "crispy", "crunchy", "very_crunchy"]

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # max 16MB upload


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    rms = np.mean(librosa.feature.rms(y=y))
    mfcc1 = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=1))
    return np.array([zcr, centroid, rolloff, rms, mfcc1])


def train_and_save_model():
    features, labels = [], []

    for idx, label in enumerate(LABELS):
        folder = os.path.join(DATASET_FOLDER, label)
        if not os.path.exists(folder):
            continue
        for file in os.listdir(folder):
            if file.lower().endswith(('.wav', '.mp3')):
                try:
                    path = os.path.join(folder, file)
                    feat = extract_features(path)
                    features.append(feat)
                    labels.append(idx)
                except Exception:
                    pass

    if not features:
        raise RuntimeError("No audio features found in dataset.")

    X = np.array(features)
    y = np.array(labels)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)

    return model


def load_or_train_model():
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return train_and_save_model()
    else:
        return train_and_save_model()


model = load_or_train_model()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        feat = extract_features(filepath).reshape(1, -1)
        pred_idx = model.predict(feat)[0]
        pred_label = LABELS[pred_idx].replace('_', ' ').title()
        print(pred_label)
        return jsonify({'prediction': pred_label})

    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


if __name__ == '__main__':
    app.run(debug=True)
