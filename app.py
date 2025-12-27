from flask import Flask, request, jsonify, send_file
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from rank_bm25 import BM25Okapi

app = Flask(__name__)

# ---------------- Load & Preprocess Dataset ----------------
df = pd.read_csv("Crop_recommendation.csv")

numerical_columns = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

scaler = MinMaxScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])

df["features"] = df[numerical_columns].astype(str).agg(" ".join, axis=1)

tokenized_corpus = [doc.split() for doc in df["features"]]
bm25 = BM25Okapi(tokenized_corpus)

# ---------------- Routes ----------------
@app.route("/")
def home():
    return send_file("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    test_input = np.array([[
        float(data["N"]),
        float(data["P"]),
        float(data["K"]),
        float(data["temperature"]),
        float(data["humidity"]),
        float(data["ph"]),
        float(data["rainfall"])
    ]])

    test_input_scaled = scaler.transform(test_input)
    test_features_str = " ".join(map(str, test_input_scaled.flatten()))
    tokenized_input = test_features_str.split()

    scores = bm25.get_scores(tokenized_input)
    top_indices = np.argsort(scores)[-3:][::-1]

    recommendations = df.iloc[top_indices]["label"].tolist()

    return jsonify({
        "recommended_crops": recommendations
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
