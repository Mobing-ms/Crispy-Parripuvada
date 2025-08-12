import os
import numpy as np
import librosa
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Automatically detect script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")

# Categories
categories = ["soft", "crispy", "crunchy", "verycrunchy"]

X = []
y = []

# Load dataset
for category in categories:
    folder_path = os.path.join(DATASET_PATH, category)
    if not os.path.exists(folder_path):
        print(f"Warning: folder not found - {folder_path}")
        continue

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        try:
            audio, sr = librosa.load(file_path, sr=None)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfccs_scaled = np.mean(mfccs.T, axis=0)
            X.append(mfccs_scaled)
            y.append(category)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

# Ensure we have data
if len(X) == 0:
    raise ValueError("No audio files found! Please check your dataset folders.")

X = np.array(X).reshape(len(X), -1)
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
model_path = os.path.join(BASE_DIR, "crunchiness_model.pkl")
joblib.dump(model, model_path)

print(f"Model saved to {model_path}")
print(f"Training accuracy: {model.score(X_train, y_train):.2f}")
print(f"Testing accuracy: {model.score(X_test, y_test):.2f}")
