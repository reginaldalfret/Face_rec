from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import base64
import json
import os
from io import BytesIO
from PIL import Image
import time
from facial_emotion_movie_recommender import FacialEmotionMovieRecommender

app = Flask(__name__)
CORS(app)

# Initialize the recommender
try:
    recommender = FacialEmotionMovieRecommender()
    print("✓ Movie recommender initialized successfully")
except Exception as e:
    print(f"⚠️ Failed to initialize movie recommender: {e}")
    recommender = None

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'recommender_available': recommender is not None
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """Analyze uploaded image for emotions and get movie recommendations"""
    if not recommender:
        return jsonify({'error': 'Movie recommender not available'}), 500
    
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode base64 image
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]  # Remove data:image/jpeg;base64, prefix
        
        image_bytes = base64.b64decode(image_data)
        
        # Convert to OpenCV format
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Detect emotions
        emotion_results = recommender.detect_emotion(image)
        
        if not emotion_results:
            return jsonify({'error': 'No faces detected in the image'}), 400
        
        # Get movie recommendations
        recommendations = recommender.recommend_movies(emotion_results)
        
        # Format response
        response_data = {
            'faces': emotion_results,
            'recommendations': []
        }
        
        # Format movie recommendations for frontend
        for movie in recommendations:
            movie_data = {
                'id': movie.get('id'),
                'title': movie.get('title', 'Unknown'),
                'year': movie.get('release_date', '')[:4] if movie.get('release_date') else 'Unknown',
                'rating': movie.get('vote_average', 0),
                'overview': movie.get('overview', 'No description available'),
                'genres': [genre['name'] for genre in movie.get('genres', [])],
                'poster_path': movie.get('poster_path')
            }
            response_data['recommendations'].append(movie_data)
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in analyze_image: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload for image analysis"""
    if not recommender:
        return jsonify({'error': 'Movie recommender not available'}), 500
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read and process the uploaded image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Detect emotions
        emotion_results = recommender.detect_emotion(image)
        
        if not emotion_results:
            return jsonify({'error': 'No faces detected in the image'}), 400
        
        # Get movie recommendations
        recommendations = recommender.recommend_movies(emotion_results)
        
        # Format response
        response_data = {
            'faces': emotion_results,
            'recommendations': []
        }
        
        # Format movie recommendations for frontend
        for movie in recommendations:
            movie_data = {
                'id': movie.get('id'),
                'title': movie.get('title', 'Unknown'),
                'year': movie.get('release_date', '')[:4] if movie.get('release_date') else 'Unknown',
                'rating': movie.get('vote_average', 0),
                'overview': movie.get('overview', 'No description available'),
                'genres': [genre['name'] for genre in movie.get('genres', [])],
                'poster_path': movie.get('poster_path')
            }
            response_data['recommendations'].append(movie_data)
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in upload_file: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/poster/<path:poster_path>')
def get_poster(poster_path):
    """Proxy endpoint to fetch movie posters from TMDB"""
    try:
        image_base_url = "https://image.tmdb.org/t/p/w500"
        full_url = f"{image_base_url}/{poster_path}"
        
        import requests
        response = requests.get(full_url, timeout=10)
        
        if response.status_code == 200:
            return response.content, 200, {'Content-Type': 'image/jpeg'}
        else:
            return jsonify({'error': 'Failed to fetch poster'}), 404
    
    except Exception as e:
        print(f"Error fetching poster: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create necessary directories
    directories = ['templates', 'static', 'static/css', 'static/js', 'static/img']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
    
    # Run the Flask app
    print("Starting Flask application...")
    print("Access the application at: http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
