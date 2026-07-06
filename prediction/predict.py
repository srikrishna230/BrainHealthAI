import tensorflow as tf
import numpy as np
import pandas as pd
import joblib

from preprocessing.feature_extraction import extract_features

# Load model
model = tf.keras.models.load_model("models/brain_model.keras")
scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/label_encoder.pkl")


def predict(df):

    # Read EEG signal
    if "EEG" in df.columns:
        signal = df["EEG"].values.astype(float)
    else:
        signal = df.iloc[:, 1].values.astype(float)

    # Feature Extraction
    features = extract_features(signal)

    features = np.array(features).reshape(1, -1)

    # Scale features
    features = scaler.transform(features)

    # Predict
    prediction = model.predict(features, verbose=0)

    index = np.argmax(prediction)

    # Get predicted class name
    label = encoder.inverse_transform([index])[0]

    confidence = float(prediction[0][index] * 100)

    probabilities = prediction[0] * 100

    neutral = float(probabilities[0])
    anxiety = float(probabilities[1])
    depression = float(probabilities[2])

    stress = round((0.7 * anxiety + 0.3 * depression), 2)

    attention = round((neutral * 0.8) + ((100 - stress) * 0.2), 2)

    happiness = round((neutral * 0.7) + ((100 - depression) * 0.3), 2)

    brain_health = round(
        (attention + happiness + (100 - stress)) / 3,
        2
    )

    return {
        "Prediction": label,
        "Confidence": confidence,
        "Probabilities": probabilities,

        "Neutral": neutral,
        "Anxiety": anxiety,
        "Depression": depression,

        "Stress": stress,
        "Attention": attention,
        "Happiness": happiness,
        "BrainHealth": brain_health
    }