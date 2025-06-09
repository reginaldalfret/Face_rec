from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from rec import FacialEmotionMovieRecommender  # Import your class

app = Flask(__name__)
recommender = FacialEmotionMovieRecommender()  # Initialize once

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    # Read image file
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Detect emotions and get movies
    emotions = recommender.detect_emotion(img)
    movies = recommender.recommend_movies(emotions)
    
    return jsonify({
        "emotions": emotions,
        "movies": movies[:3]  # Return top 3 movies
    })

if __name__ == "__main__":
    app.run(debug=True)