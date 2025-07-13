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
from datetime import datetime
from bson import ObjectId
import gridfs
from pymongo import MongoClient
from facial_emotion_movie_recommender import FacialEmotionMovieRecommender

app = Flask(__name__)
CORS(app)

# MongoDB Configuration
MONGODB_URI = "mongodb+srv://facerec:xbwb9c8NwdGxhX9c@cluster0.dqjvi36.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DATABASE_NAME = "facial_emotion_db"

# Initialize MongoDB connection
try:
    client = MongoClient(MONGODB_URI)
    db = client[DATABASE_NAME]
    
    # Collections
    analysis_collection = db.analyses
    users_collection = db.users
    
    # GridFS for storing images
    fs = gridfs.GridFS(db)
    
    # Test connection
    client.admin.command('ping')
    print("✓ MongoDB connected successfully")
    
except Exception as e:
    print(f"⚠️ MongoDB connection failed: {e}")
    client = None
    db = None
    fs = None

# Initialize the recommender
try:
    recommender = FacialEmotionMovieRecommender()
    print("✓ Movie recommender initialized successfully")
except Exception as e:
    print(f"⚠️ Failed to initialize movie recommender: {e}")
    recommender = None

class DatabaseManager:
    """Handle database operations"""
    
    @staticmethod
    def store_image(image_data, filename):
        """Store image in GridFS and return file_id"""
        try:
            if fs is None:
                return None
            
            # Store image with metadata
            file_id = fs.put(
                image_data,
                filename=filename,
                upload_date=datetime.utcnow(),
                content_type="image/jpeg"
            )
            return str(file_id)
        except Exception as e:
            print(f"Error storing image: {e}")
            return None
    
    @staticmethod
    def get_image(file_id):
        """Retrieve image from GridFS"""
        try:
            if fs is None:
                return None
            
            grid_out = fs.get(ObjectId(file_id))
            return grid_out.read()
        except Exception as e:
            print(f"Error retrieving image: {e}")
            return None
    
    @staticmethod
    def store_analysis(analysis_data):
        """Store analysis metadata in MongoDB"""
        try:
            if analysis_collection is None:
                return None
            
            # Add timestamp
            analysis_data['created_at'] = datetime.utcnow()
            analysis_data['updated_at'] = datetime.utcnow()
            
            result = analysis_collection.insert_one(analysis_data)
            return str(result.inserted_id)
        except Exception as e:
            print(f"Error storing analysis: {e}")
            return None
    
    @staticmethod
    def get_analysis(analysis_id):
        """Retrieve analysis by ID"""
        try:
            if analysis_collection is None:
                return None
            
            return analysis_collection.find_one({"_id": ObjectId(analysis_id)})
        except Exception as e:
            print(f"Error retrieving analysis: {e}")
            return None
    
    @staticmethod
    def get_user_analyses(user_id=None, limit=10):
        """Get recent analyses for a user or all analyses"""
        try:
            if analysis_collection is None:
                return []
            
            query = {"user_id": user_id} if user_id else {}
            cursor = analysis_collection.find(query).sort("created_at", -1).limit(limit)
            
            analyses = []
            for doc in cursor:
                doc['_id'] = str(doc['_id'])
                analyses.append(doc)
            
            return analyses
        except Exception as e:
            print(f"Error retrieving user analyses: {e}")
            return []
    
    @staticmethod
    def store_user_session(session_data):
        """Store user session data"""
        try:
            if users_collection is None:
                return None
            
            session_data['created_at'] = datetime.utcnow()
            session_data['last_activity'] = datetime.utcnow()
            
            result = users_collection.insert_one(session_data)
            return str(result.inserted_id)
        except Exception as e:
            print(f"Error storing user session: {e}")
            return None

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'recommender_available': recommender is not None,
        'database_connected': db is not None
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
        
        # Store image in database
        image_id = None
        if db is not None:
            filename = f"analysis_{int(time.time())}.jpg"
            image_id = DatabaseManager.store_image(image_bytes, filename)
        
        # Detect emotions
        emotion_results = recommender.detect_emotion(image)
        
        # Debug: Print emotion results structure
        print(f"DEBUG: Emotion results structure: {emotion_results}")
        
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
        
        # Store analysis metadata in database
        if db is not None:
            # Get primary emotion safely
            primary_emotion = None
            confidence_score = 0
            
            if emotion_results:
                first_result = emotion_results[0]
                # Handle different possible structures
                if isinstance(first_result, dict):
                    primary_emotion = first_result.get('emotion') or first_result.get('dominant_emotion')
                    confidence_score = first_result.get('confidence', 0)
                    
                    # If no emotion key, find the highest confidence emotion
                    if not primary_emotion and 'emotions' in first_result:
                        emotions = first_result['emotions']
                        if emotions:
                            primary_emotion = max(emotions, key=emotions.get)
                            confidence_score = emotions[primary_emotion]
            
            analysis_metadata = {
                'user_id': data.get('user_id'),  # Optional user ID from frontend
                'image_id': image_id,
                'emotion_results': emotion_results,
                'recommendations': response_data['recommendations'],
                'analysis_type': 'webcam_capture',
                'face_count': len(emotion_results),
                'primary_emotion': primary_emotion,
                'confidence_score': confidence_score
            }
            
            analysis_id = DatabaseManager.store_analysis(analysis_metadata)
            response_data['analysis_id'] = analysis_id
        
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
        
        # Store image in database
        image_id = None
        if db is not None:
            filename = f"upload_{int(time.time())}_{file.filename}"
            image_id = DatabaseManager.store_image(image_bytes, filename)
        
        # Detect emotions
        emotion_results = recommender.detect_emotion(image)
        
        # Debug: Print emotion results structure
        print(f"DEBUG: Emotion results structure: {emotion_results}")
        
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
        
        # Store analysis metadata in database
        if db is not None:
            # Get primary emotion safely
            primary_emotion = None
            confidence_score = 0
            
            if emotion_results:
                first_result = emotion_results[0]
                # Handle different possible structures
                if isinstance(first_result, dict):
                    primary_emotion = first_result.get('emotion') or first_result.get('dominant_emotion')
                    confidence_score = first_result.get('confidence', 0)
                    
                    # If no emotion key, find the highest confidence emotion
                    if not primary_emotion and 'emotions' in first_result:
                        emotions = first_result['emotions']
                        if emotions:
                            primary_emotion = max(emotions, key=emotions.get)
                            confidence_score = emotions[primary_emotion]
            
            analysis_metadata = {
                'user_id': request.form.get('user_id'),  # Optional user ID
                'image_id': image_id,
                'emotion_results': emotion_results,
                'recommendations': response_data['recommendations'],
                'analysis_type': 'file_upload',
                'original_filename': file.filename,
                'face_count': len(emotion_results),
                'primary_emotion': primary_emotion,
                'confidence_score': confidence_score
            }
            
            analysis_id = DatabaseManager.store_analysis(analysis_metadata)
            response_data['analysis_id'] = analysis_id
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in upload_file: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/history')
def get_analysis_history():
    """Get analysis history for a user"""
    try:
        user_id = request.args.get('user_id')
        limit = int(request.args.get('limit', 10))
        
        if db is None:
            return jsonify({'error': 'Database not available'}), 500
        
        analyses = DatabaseManager.get_user_analyses(user_id, limit)
        
        return jsonify({
            'analyses': analyses,
            'count': len(analyses)
        })
        
    except Exception as e:
        print(f"Error retrieving history: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analysis/<analysis_id>')
def get_analysis_details(analysis_id):
    """Get detailed analysis information"""
    try:
        if db is None:
            return jsonify({'error': 'Database not available'}), 500
        
        analysis = DatabaseManager.get_analysis(analysis_id)
        if not analysis:
            return jsonify({'error': 'Analysis not found'}), 404
        
        # Convert ObjectId to string for JSON serialization
        analysis['_id'] = str(analysis['_id'])
        
        return jsonify(analysis)
        
    except Exception as e:
        print(f"Error retrieving analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/image/<image_id>')
def get_stored_image(image_id):
    """Retrieve stored image from database"""
    try:
        if db is None:
            return jsonify({'error': 'Database not available'}), 500
        
        image_data = DatabaseManager.get_image(image_id)
        if not image_data:
            return jsonify({'error': 'Image not found'}), 404
        
        return image_data, 200, {'Content-Type': 'image/jpeg'}
        
    except Exception as e:
        print(f"Error retrieving image: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/user/session', methods=['POST'])
def create_user_session():
    """Create a new user session"""
    try:
        if db is None:
            return jsonify({'error': 'Database not available'}), 500
        
        data = request.get_json()
        session_data = {
            'user_agent': request.headers.get('User-Agent'),
            'ip_address': request.remote_addr,
            'session_data': data
        }
        
        session_id = DatabaseManager.store_user_session(session_data)
        
        return jsonify({
            'session_id': session_id,
            'status': 'created'
        })
        
    except Exception as e:
        print(f"Error creating session: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def get_statistics():
    """Get application statistics"""
    try:
        if db is None:
            return jsonify({'error': 'Database not available'}), 500
        
        total_analyses = analysis_collection.count_documents({})
        total_users = users_collection.count_documents({})
        
        # Get emotion distribution
        pipeline = [
            {"$group": {"_id": "$primary_emotion", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        emotion_stats = list(analysis_collection.aggregate(pipeline))
        
        return jsonify({
            'total_analyses': total_analyses,
            'total_users': total_users,
            'emotion_distribution': emotion_stats
        })
        
    except Exception as e:
        print(f"Error retrieving statistics: {e}")
        return jsonify({'error': str(e)}), 500

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
    print("MongoDB connection:", "✓ Connected" if db is not None else "✗ Failed")
    app.run(debug=True, host='0.0.0.0', port=5000)
