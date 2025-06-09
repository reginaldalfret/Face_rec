from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
import base64
import json
import os
from io import BytesIO
from PIL import Image
import time

# Import your existing class
from rec import FacialEmotionMovieRecommender

app = Flask(__name__)
recommender = FacialEmotionMovieRecommender()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_webcam', methods=['POST'])
def process_webcam():
    try:
        # Get the image data from the request
        image_data = request.json.get('image')
        if not image_data:
            return jsonify({'error': 'No image data received'}), 400
        
        # Remove the data URL prefix
        image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Convert PIL Image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Process the image with your recommender
        emotion_results = recommender.detect_emotion(image_cv)
        
        if not emotion_results:
            return jsonify({'error': 'No faces detected'}), 400
        
        # Get movie recommendations
        recommendations = recommender.recommend_movies(emotion_results)
        
        # Prepare response data
        response_data = {
            'emotions': emotion_results,
            'recommendations': []
        }
        
        # Add movie data
        for movie in recommendations[:5]:  # Limit to 5 recommendations
            movie_data = {
                'title': movie.get('title', 'Unknown'),
                'year': movie.get('release_date', '')[:4] if movie.get('release_date') else 'Unknown',
                'rating': movie.get('vote_average', 'N/A'),
                'overview': movie.get('overview', 'No description available'),
                'genres': [genre['name'] for genre in movie.get('genres', [])],
                'poster_path': movie.get('poster_path', None)
            }
            
            response_data['recommendations'].append(movie_data)
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/process_upload', methods=['POST'])
def process_upload():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Read and process the uploaded image
        image_bytes = file.read()
        image = Image.open(BytesIO(image_bytes))
        
        # Convert PIL Image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Process the image with your recommender
        emotion_results = recommender.detect_emotion(image_cv)
        
        if not emotion_results:
            return jsonify({'error': 'No faces detected'}), 400
        
        # Get movie recommendations
        recommendations = recommender.recommend_movies(emotion_results)
        
        # Prepare response data
        response_data = {
            'emotions': emotion_results,
            'recommendations': []
        }
        
        # Add movie data
        for movie in recommendations[:5]:  # Limit to 5 recommendations
            movie_data = {
                'title': movie.get('title', 'Unknown'),
                'year': movie.get('release_date', '')[:4] if movie.get('release_date') else 'Unknown',
                'rating': movie.get('vote_average', 'N/A'),
                'overview': movie.get('overview', 'No description available'),
                'genres': [genre['name'] for genre in movie.get('genres', [])],
                'poster_path': movie.get('poster_path', None)
            }
            
            response_data['recommendations'].append(movie_data)
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_poster/<path:poster_path>')
def get_poster(poster_path):
    try:
        image_base_url = "https://image.tmdb.org/t/p/w500"
        full_url = f"{image_base_url}/{poster_path}"
        
        import requests
        response = requests.get(full_url)
        
        if response.status_code == 200:
            return Response(
                response.content,
                mimetype='image/jpeg',
                headers={'Content-Disposition': 'inline'}
            )
        else:
            return jsonify({'error': 'Failed to fetch poster'}), 404
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create necessary directories
    if not os.path.exists('movie_posters'):
        os.makedirs('movie_posters')
    
    # Run the Flask app
    app.run(debug=True)