#!/usr/bin/env python
# coding: utf-8

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import Image
import time
import os
import json
import random
from datetime import datetime

class FacialEmotionMovieRecommender:
    def __init__(self):
        # Initialize face detector using OpenCV's Haar Cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Haar cascade file not found at {cascade_path}")
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Initialize emotion detection
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # TMDB API credentials
        self.api_key = "58f44feb6510bf8843a41176a48d0489"
        self.access_token = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI1OGY0NGZlYjY1MTBiZjg4NDNhNDExNzZhNDhkMDQ4OSIsIm5iZiI6MTc0MjM5NjYyNS40OTQwMDAyLCJzdWIiOiI2N2RhZGNkMWU1NGU5MjJkNzI2Y2MzNmMiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.6UCVO3gAwnLUnKMNO0NPm2wMNrroe_ScC798uwoAjgk"
        self.base_url = "https://api.themoviedb.org/3"
        self.image_base_url = "https://image.tmdb.org/t/p/w500"
        
        # Test API connection
        self.test_api_connection()
        
        # Load or create genre mappings
        self.load_genre_data()
        
        # Emotion to genre mapping
        self.emotion_genre_map = {
            'angry': [28, 80, 18],  # Action, Crime, Drama
            'disgust': [27, 53],    # Horror, Thriller
            'fear': [27, 53],       # Horror, Thriller
            'happy': [35, 16, 10402, 10749],  # Comedy, Animation, Music, Romance
            'sad': [18, 10749],     # Drama, Romance
            'surprise': [878, 28, 53],  # Science Fiction, Action, Thriller
            'neutral': [18, 878, 80]    # Drama, Science Fiction, Crime
        }

        # Create necessary directories
        self.create_directories()
            
        # Cache for movies to avoid repeated API calls
        self.movie_cache = {}
        
    def create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = ['movie_posters', 'output_images', 'temp']
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")
        
    def test_api_connection(self):
        """Test connection to TMDB API"""
        print("Testing connection to TMDB API...")
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json;charset=utf-8"
        }
        
        try:
            response = requests.get(
                f"{self.base_url}/configuration",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                print("✓ Successfully connected to TMDB API")
                self.config = response.json()
                return True
            else:
                print(f"⚠️ API connection failed with status code: {response.status_code}")
                print(f"Error message: {response.text}")
                return False
        except Exception as e:
            print(f"⚠️ Error connecting to TMDB API: {e}")
            return False
    
    def load_genre_data(self):
        """Load genre data from TMDB or cache file"""
        cache_file = "genre_cache.json"
        
        # Try to load from cache first
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.genres = json.load(f)
                print("✓ Loaded genre data from cache")
                return
            except Exception as e:
                print(f"Error loading genre cache: {e}")
        
        # If cache doesn't exist or is corrupted, fetch from API
        print("Fetching genre data from TMDB API...")
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json;charset=utf-8"
        }
        
        try:
            response = requests.get(
                f"{self.base_url}/genre/movie/list",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                self.genres = response.json()
                # Save to cache
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.genres, f, indent=2)
                print("✓ Successfully fetched and cached genre data")
            else:
                print(f"⚠️ Failed to fetch genre data: {response.status_code}")
                self._use_fallback_genres()
        except Exception as e:
            print(f"⚠️ Error fetching genre data: {e}")
            self._use_fallback_genres()
    
    def _use_fallback_genres(self):
        """Use fallback genre data when API is unavailable"""
        self.genres = {
            "genres": [
                {"id": 28, "name": "Action"},
                {"id": 12, "name": "Adventure"},
                {"id": 16, "name": "Animation"},
                {"id": 35, "name": "Comedy"},
                {"id": 80, "name": "Crime"},
                {"id": 18, "name": "Drama"},
                {"id": 27, "name": "Horror"},
                {"id": 10749, "name": "Romance"},
                {"id": 878, "name": "Science Fiction"},
                {"id": 53, "name": "Thriller"}
            ]
        }
        print("✓ Using fallback genre data")
    
    def get_genre_name(self, genre_id):
        """Convert genre ID to name"""
        for genre in self.genres["genres"]:
            if genre["id"] == genre_id:
                return genre["name"]
        return "Unknown"
    
    def get_genre_id(self, genre_name):
        """Convert genre name to ID"""
        for genre in self.genres["genres"]:
            if genre["name"].lower() == genre_name.lower():
                return genre["id"]
        return None

    def capture_image(self):
        """Capture an image from the webcam with improved error handling"""
        print("Initializing camera...")

        cap = None
        camera_found = False

        # Try different camera indices with better error handling
        for camera_index in range(5):  # Try indices 0-4
            try:
                if cap is not None:
                    cap.release()  # Release previous camera if any

                cap = cv2.VideoCapture(camera_index)

                # Check if camera opened successfully
                if not cap.isOpened():
                    continue

                # Try to read a test frame
                ret, test_frame = cap.read()
                if not ret or test_frame is None or test_frame.size == 0:
                    continue

                print(f"✓ Camera found and working at index {camera_index}")
                camera_found = True
                break

            except Exception as e:
                print(f"Error with camera at index {camera_index}: {e}")
                continue

        if not camera_found:
            print("Error: Could not find any working camera. Please check your connections or permissions.")
            return None

        try:
            # Camera found, now configure it
            print("Camera connected successfully.")

            # Set properties - try different resolutions if having issues
            resolutions = [(640, 480), (800, 600), (1280, 720)]
            for width, height in resolutions:
                try:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    print(f"Camera resolution set to: {actual_width}x{actual_height}")
                    break
                except Exception as e:
                    print(f"Failed to set resolution {width}x{height}: {e}")

            # Give camera time to properly initialize
            print("Warming up camera...")
            warm_up_frames = 10
            for i in range(warm_up_frames):
                ret, _ = cap.read()  # Discard frames during warm-up
                time.sleep(0.1)

            print("Taking your picture in:")
            for i in range(3, 0, -1):
                print(f"{i}...")
                # Read and discard frames to flush buffer
                for _ in range(3):
                    cap.read()
                time.sleep(1)

            print("Say cheese! 📸")

            # Try multiple captures to get the best frame
            frames = []
            max_attempts = 5

            for attempt in range(max_attempts):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0 and not np.all(frame == 0):
                    frames.append(frame)
                    print(f"Frame {attempt+1} captured")
                else:
                    print(f"Failed to capture frame {attempt+1}")
                time.sleep(0.2)

            if not frames:
                print("Failed to capture any valid images. Please check your camera and lighting.")
                return None

            # Use the last frame (usually the best one after camera has adjusted)
            best_frame = frames[-1]

            # Save the captured image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"output_images/captured_image_{timestamp}.jpg"
            cv2.imwrite(filename, best_frame)

            # Display the captured image
            plt.figure(figsize=(8, 6))
            plt.imshow(cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB))
            plt.title("Captured Image")
            plt.axis('off')
            plt.savefig(f"output_images/captured_display_{timestamp}.png", bbox_inches='tight', dpi=150)
            plt.close()

            return best_frame

        finally:
            # Always release the camera
            if cap is not None:
                cap.release()

    def upload_image(self, image_path):
        """Load an image from a file path"""
        try:
            if not os.path.exists(image_path):
                print(f"Image file not found: {image_path}")
                return None

            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image from {image_path}")
                return None

            # Save a copy for reference
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"output_images/uploaded_display_{timestamp}.png"

            # Display the loaded image
            plt.figure(figsize=(8, 6))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title("Uploaded Image")
            plt.axis('off')
            plt.savefig(filename, bbox_inches='tight', dpi=150)
            plt.close()

            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    def simple_emotion_detection(self, face_roi):
        """Simple rule-based emotion detection based on facial features"""
        try:
            # Convert to grayscale
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
            
            # Calculate basic statistics
            mean_intensity = np.mean(gray_face)
            std_intensity = np.std(gray_face)
            
            # Simple heuristics based on image properties
            emotions = {
                'angry': 0.1,
                'disgust': 0.05,
                'fear': 0.05,
                'happy': 0.3,  # Default to somewhat happy
                'sad': 0.1,
                'surprise': 0.1,
                'neutral': 0.3
            }
            
            # Adjust based on image brightness (brighter = happier assumption)
            if mean_intensity > 120:
                emotions['happy'] += 0.2
                emotions['neutral'] -= 0.1
            elif mean_intensity < 80:
                emotions['sad'] += 0.2
                emotions['neutral'] -= 0.1
            
            # Add some randomness to make it more interesting
            emotion_keys = list(emotions.keys())
            boost_emotion = random.choice(emotion_keys)
            emotions[boost_emotion] += 0.1
            
            # Normalize probabilities
            total = sum(emotions.values())
            if total > 0:
                for emotion in emotions:
                    emotions[emotion] /= total
                    
            return emotions
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            # Return default emotions if error occurs
            return {
                'happy': 0.4,
                'neutral': 0.3,
                'surprise': 0.1,
                'sad': 0.1,
                'angry': 0.05,
                'fear': 0.03,
                'disgust': 0.02
            }

    def detect_emotion(self, image):
        """Detect emotions in the given image using OpenCV face detection"""
        try:
            if image is None:
                print("No image provided for emotion detection")
                return None

            # Create a copy of the image for processing
            image_copy = image.copy()
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            if len(faces) == 0:
                print("No faces detected in the image.")
                return None

            # Process each detected face
            results = []
            for i, (x, y, w, h) in enumerate(faces):
                # Draw rectangle around the face
                cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Extract face region
                face_roi = image_copy[y:y+h, x:x+w]
                
                # Simple emotion detection
                emotions_prob = self.simple_emotion_detection(face_roi)
                dominant_emotion = max(emotions_prob.items(), key=lambda x: x[1])
                
                # Add text with dominant emotion
                text = f"{dominant_emotion[0]}: {dominant_emotion[1]:.2f}"
                cv2.putText(image_copy, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Store results
                results.append({
                    'face_id': i + 1,
                    'box': [int(x), int(y), int(w), int(h)],
                    'emotions': emotions_prob,
                    'dominant_emotion': dominant_emotion[0]
                })

            # Save the image with detected faces and emotions
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"output_images/emotion_detection_{timestamp}.png"
            
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
            plt.title("Detected Emotions")
            plt.axis('off')
            plt.savefig(filename, bbox_inches='tight', dpi=150)
            plt.close()

            return results

        except Exception as e:
            print(f"Error detecting emotions: {e}")
            return None

    def fetch_movies_by_genre(self, genre_ids, page=1, min_rating=7.0):
        """Fetch movies from TMDB API based on genre IDs"""
        # Create a cache key from the genre IDs and page
        cache_key = f"genres_{'-'.join(map(str, genre_ids))}_page_{page}_rating_{min_rating}"
        
        # Check if we have this query cached
        if cache_key in self.movie_cache:
            print(f"Using cached results for {cache_key}")
            return self.movie_cache[cache_key]
        
        print(f"Fetching movies for genres: {[self.get_genre_name(g) for g in genre_ids]}")
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json;charset=utf-8"
        }
        
        try:
            # Convert genre IDs to comma-separated string
            genre_str = ",".join(map(str, genre_ids))
            
            # Make API request
            response = requests.get(
                f"{self.base_url}/discover/movie",
                headers=headers,
                params={
                    "with_genres": genre_str,
                    "sort_by": "vote_average.desc",
                    "vote_count.gte": 100,  # Only include movies with at least 100 votes
                    "vote_average.gte": min_rating,  # Only include movies with rating >= min_rating
                    "page": page,
                    "language": "en-US"
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                # Cache the results
                self.movie_cache[cache_key] = data
                return data
            else:
                print(f"⚠️ API request failed: {response.status_code}")
                print(f"Error message: {response.text}")
                return None
        except Exception as e:
            print(f"⚠️ Error fetching movies: {e}")
            return None

    def get_movie_details(self, movie_id):
        """Get detailed information about a specific movie"""
        # Check if we have this movie cached
        cache_key = f"movie_{movie_id}"
        if cache_key in self.movie_cache:
            return self.movie_cache[cache_key]
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json;charset=utf-8"
        }
        
        try:
            response = requests.get(
                f"{self.base_url}/movie/{movie_id}",
                headers=headers,
                params={"language": "en-US", "append_to_response": "keywords,credits"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                # Cache the results
                self.movie_cache[cache_key] = data
                return data
            else:
                print(f"⚠️ Failed to get movie details: {response.status_code}")
                return None
        except Exception as e:
            print(f"⚠️ Error getting movie details: {e}")
            return None

    def map_movie_to_emotions(self, movie_data):
        """Map a movie to emotions based on its genres and keywords"""
        if not movie_data:
            return []
        
        # Get the movie's genres
        genre_ids = [genre["id"] for genre in movie_data.get("genres", [])]
        
        # Map genres to emotions
        emotions = []
        for emotion, genre_list in self.emotion_genre_map.items():
            if any(genre_id in genre_list for genre_id in genre_ids):
                emotions.append(emotion)
        
        # Add additional mapping based on keywords if available
        if "keywords" in movie_data and "keywords" in movie_data["keywords"]:
            keywords = [kw["name"].lower() for kw in movie_data["keywords"]["keywords"]]
            
            # Check for specific keywords
            keyword_emotion_map = {
                "happy": ["comedy", "funny", "humor", "laugh"],
                "sad": ["sad", "tragedy", "death", "grief"],
                "fear": ["scary", "horror", "terror", "suspense"],
                "angry": ["action", "fight", "violence", "war"],
                "surprise": ["twist", "surprise", "unexpected", "shocking"]
            }
            
            for emotion, emotion_keywords in keyword_emotion_map.items():
                if any(kw in keywords for kw in emotion_keywords):
                    if emotion not in emotions:
                        emotions.append(emotion)
        
        return emotions

    def download_poster(self, movie_data):
        """Download and save movie poster"""
        if not movie_data or "poster_path" not in movie_data or not movie_data["poster_path"]:
            return None
            
        try:
            # Clean the title for filename
            movie_title = movie_data["title"]
            clean_title = ''.join(c if c.isalnum() else '_' for c in movie_title)
            filename = f"movie_posters/{clean_title}.jpg"

            # Check if we already downloaded this poster
            if os.path.exists(filename):
                # Load the existing image
                return plt.imread(filename)

            # Construct the full poster URL
            poster_url = f"{self.image_base_url}{movie_data['poster_path']}"
            
            # Try to download the poster
            response = requests.get(poster_url, timeout=10)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                img.save(filename)
                return np.array(img)
            else:
                print(f"Failed to download poster for {movie_title}: Status code {response.status_code}")
                return None
        except Exception as e:
            print(f"Error downloading poster: {e}")
            return None

    def recommend_movies(self, emotion_results, num_recommendations=5):
        """Recommend movies based on detected emotions"""
        if not emotion_results:
            print("No emotion data available for recommendations.")
            return []

        try:
            # Collect all detected emotions with their probabilities
            all_emotions = {}
            for face in emotion_results:
                for emotion, prob in face['emotions'].items():
                    if emotion in all_emotions:
                        all_emotions[emotion] += prob
                    else:
                        all_emotions[emotion] = prob

            # Normalize emotion scores
            total = sum(all_emotions.values())
            if total > 0:
                for emotion in all_emotions:
                    all_emotions[emotion] /= total

            # Sort emotions by probability
            sorted_emotions = sorted(all_emotions.items(), key=lambda x: x[1], reverse=True)
            print("\n🔍 Detected Emotions:")
            for emotion, prob in sorted_emotions:
                print(f"  - {emotion.capitalize()}: {prob:.2f}")

            # Take the top 2 emotions for recommendations
            top_emotions = [emotion for emotion, _ in sorted_emotions[:2]]

            # Get suitable genres based on emotions
            suitable_genre_ids = []
            for emotion in top_emotions:
                if emotion in self.emotion_genre_map:
                    suitable_genre_ids.extend(self.emotion_genre_map[emotion])

            # Remove duplicates
            suitable_genre_ids = list(set(suitable_genre_ids))
            genre_names = [self.get_genre_name(genre_id) for genre_id in suitable_genre_ids]

            print(f"\n🎭 Based on your emotions, we think you might enjoy these genres: {', '.join(genre_names)}")

            # Fetch movies by genres
            movies_data = self.fetch_movies_by_genre(suitable_genre_ids)
            
            if not movies_data or "results" not in movies_data or len(movies_data["results"]) == 0:
                print("⚠️ No movies found for these genres. Trying with broader criteria...")
                movies_data = self.fetch_movies_by_genre(suitable_genre_ids, min_rating=6.0)
            
            if not movies_data or "results" not in movies_data or len(movies_data["results"]) == 0:
                print("⚠️ Still no movies found. Showing popular movies instead.")
                # Fetch popular movies instead
                try:
                    headers = {
                        "Authorization": f"Bearer {self.access_token}",
                        "Content-Type": "application/json;charset=utf-8"
                    }
                    response = requests.get(
                        f"{self.base_url}/movie/popular",
                        headers=headers,
                        params={"language": "en-US", "page": 1},
                        timeout=10
                    )
                    if response.status_code == 200:
                        movies_data = response.json()
                    else:
                        print(f"⚠️ Failed to fetch popular movies: {response.status_code}")
                        return []
                except Exception as e:
                    print(f"⚠️ Error fetching popular movies: {e}")
                    return []

            # Process the results
            movies = movies_data["results"]
            
            # Get complete movie details and map to emotions
            detailed_movies = []
            for movie in movies[:min(10, len(movies))]:  # Limit to avoid too many API calls
                movie_id = movie["id"]
                movie_details = self.get_movie_details(movie_id)
                if movie_details:
                    movie_details["emotion_tags"] = self.map_movie_to_emotions(movie_details)
                    detailed_movies.append(movie_details)
            
            # Sort movies by relevance to detected emotions
            movie_scores = []
            for movie in detailed_movies:
                score = 0
                for emotion in top_emotions:
                    if emotion in movie["emotion_tags"]:
                        score += 1
                movie_scores.append((movie, score))
            
            # Sort by score (descending) and then by rating (descending)
            movie_scores.sort(key=lambda x: (x[1], x[0].get("vote_average", 0)), reverse=True)
            
            # Get the top recommendations
            recommendations = [movie for movie, _ in movie_scores[:num_recommendations]]
            
            if not recommendations:
                print("⚠️ No suitable movies found.")
                return []
                
            # Display recommendations
            print(f"\n🎬 Here are your personalized movie recommendations:")

            for i, movie in enumerate(recommendations, 1):
                # Get genre names
                genre_names = [genre["name"] for genre in movie.get("genres", [])]
                genre_str = ", ".join(genre_names) if genre_names else "Unknown"
                
                # Get year from release date
                year = movie.get("release_date", "")[:4] if movie.get("release_date") else "Unknown"
                
                # Display movie information
                print(f"\n{i}. {movie['title']} ({year}) - {genre_str} - Rating: {movie.get('vote_average', 'N/A')}/10")
                print(f"   {movie.get('overview', 'No description available')}")

            # Create a visual recommendation display
            self._display_movie_recommendations(recommendations)

            return recommendations

        except Exception as e:
            print(f"Error in recommend_movies: {e}")
            return []

    def _display_movie_recommendations(self, recommendations):
        """Display movie recommendations with downloaded posters"""
        n_movies = len(recommendations)
        if n_movies == 0:
            return

        try:
            # Download posters in advance
            posters = []
            titles = []

            print("\nDownloading movie posters...")
            for movie in recommendations:
                title = movie['title']
                year = movie.get("release_date", "")[:4] if movie.get("release_date") else "Unknown"
                titles.append(f"{title}\n({year})\nRating: {movie.get('vote_average', 'N/A')}")

                # Try to download the poster
                poster_img = self.download_poster(movie)

                if poster_img is not None:
                    posters.append(poster_img)
                else:
                    # Create a placeholder if download fails
                    placeholder = np.ones((400, 300, 3)) * 0.8  # Light gray placeholder
                    posters.append(placeholder)

            # Display the posters
            fig, axes = plt.subplots(1, min(n_movies, 5), figsize=(15, 10))
            if n_movies == 1:
                axes = [axes]
            elif n_movies == 0:
                return

            for i in range(min(n_movies, 5)):
                if i < len(axes):
                    axes[i].imshow(posters[i])
                    axes[i].set_title(titles[i], fontsize=10)
                    axes[i].axis('off')

            plt.tight_layout()
            
            # Save the recommendations display
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"output_images/movie_recommendations_{timestamp}.png"
            plt.savefig(filename, bbox_inches='tight', dpi=150)
            plt.close()
            
            print(f"Movie recommendations saved to: {filename}")

        except Exception as e:
            print(f"Error displaying movie recommendations: {e}")

    def run(self):
        """Run the full recommendation pipeline with user interface"""
        print("=" * 60)
        print("🎭 FACIAL EMOTION MOVIE RECOMMENDER 🎬")
        print("=" * 60)
        print("\nThis app will analyze your facial expression and recommend movies based on your mood!")
        print("Now using real-time data from The Movie Database (TMDB) API!")
        print("Note: Using simplified emotion detection")

        while True:
            print("\nOptions:")
            print("1. Capture image from webcam")
            print("2. Upload an image")
            print("3. Use sample image")
            print("4. Quit")

            try:
                choice = input("\nEnter your choice (1-4): ").strip()

                if choice == '1':
                    image = self.capture_image()
                elif choice == '2':
                    image_path = input("Enter the path to your image file: ").strip()
                    image = self.upload_image(image_path)
                elif choice == '3':
                    # Use a sample image showing a happy face
                    print("Using a sample image (simulated for this example)")
                    # Create a black image with a simple happy face
                    image = np.zeros((300, 300, 3), dtype=np.uint8)
                    # Draw a simple face (this is just for demonstration)
                    cv2.circle(image, (150, 150), 100, (200, 200, 200), -1)  # Face
                    cv2.circle(image, (110, 130), 15, (255, 255, 255), -1)  # Left eye
                    cv2.circle(image, (190, 130), 15, (255, 255, 255), -1)  # Right eye
                    cv2.ellipse(image, (150, 180), (50, 20), 0, 0, 180, (255, 255, 255), -1)  # Smile

                    # Save the sample image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"output_images/sample_image_{timestamp}.png"
                    
                    plt.figure(figsize=(5, 5))
                    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    plt.title("Sample Image")
                    plt.axis('off')
                    plt.savefig(filename, bbox_inches='tight', dpi=150)
                    plt.close()

                    # For the sample image, simulate emotion detection results
                    emotion_results = [{
                        'face_id': 1,
                        'box': [50, 50, 200, 200],
                        'emotions': {'happy': 0.7, 'neutral': 0.2, 'surprise': 0.1},
                        'dominant_emotion': 'happy'
                    }]

                    print("\nDetected Emotions (simulated):")
                    print("- Happy: 70%")
                    print("- Neutral: 20%")
                    print("- Surprise: 10%")

                    self.recommend_movies(emotion_results)
                    continue
                elif choice == '4':
                    print("\nThank you for using the Facial Emotion Movie Recommender! Goodbye!")
                    break
                else:
                    print("Invalid choice. Please try again.")
                    continue

                if image is not None:
                    emotion_results = self.detect_emotion(image)
                    if emotion_results:
                        self.recommend_movies(emotion_results)
                    else:
                        print("Could not detect emotions in the image. Please try again with a clearer image.")
                else:
                    print("Failed to load image. Please try again.")

            except KeyboardInterrupt:
                print("\n\nProgram interrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"An error occurred: {e}")
                print("Please try again.")

            print("\n" + "-" * 60)

# Main execution function
def run_movie_recommender():
    """Main function to run the movie recommender"""
    try:
        recommender = FacialEmotionMovieRecommender()
        recommender.run()
    except Exception as e:
        print(f"Failed to initialize the movie recommender: {e}")
        print("Please check your dependencies and try again.")

# Run the application
if __name__ == "__main__":
    run_movie_recommender()
