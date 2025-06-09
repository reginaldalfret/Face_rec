document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const webcamElement = document.getElementById('webcam');
    const canvasElement = document.getElementById('canvas');
    const captureBtn = document.getElementById('capture-btn');
    const retryBtn = document.getElementById('retry-btn');
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const fileNameDisplay = document.getElementById('file-name');
    const loadingElement = document.getElementById('loading');
    const resultsElement = document.getElementById('results');
    const emotionsListElement = document.getElementById('emotions-list');
    const recommendationsElement = document.getElementById('recommendations');
    const errorMessageElement = document.getElementById('error-message');
    const errorDetailsElement = document.getElementById('error-details');
    
    let stream = null;
    let emotionsChart = null;
    
    // Tab switching
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.getAttribute('data-tab');
            
            // Update active tab button
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Update active tab content
            tabContents.forEach(content => {
                content.classList.remove('active');
                if (content.id === `${tabId}-tab`) {
                    content.classList.add('active');
                }
            });
            
            // Start or stop webcam based on active tab
            if (tabId === 'webcam') {
                startWebcam();
            } else {
                stopWebcam();
            }
            
            // Reset UI
            resetUI();
        });
    });
    
    // Start webcam
    async function startWebcam() {
        try {
            const constraints = {
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                }
            };
            
            stream = await navigator.mediaDevices.getUserMedia(constraints);
            webcamElement.srcObject = stream;
            
            // Show capture button, hide retry button
            captureBtn.style.display = 'block';
            retryBtn.style.display = 'none';
        } catch (error) {
            console.error('Error accessing webcam:', error);
            showError('Could not access webcam. Please make sure you have granted camera permissions.');
        }
    }
    
    // Stop webcam
    function stopWebcam() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            webcamElement.srcObject = null;
            stream = null;
        }
    }
    
    // Capture image from webcam
    captureBtn.addEventListener('click', () => {
        // Set canvas dimensions to match video
        canvasElement.width = webcamElement.videoWidth;
        canvasElement.height = webcamElement.videoHeight;
        
        // Draw video frame to canvas
        const context = canvasElement.getContext('2d');
        context.drawImage(webcamElement, 0, 0, canvasElement.width, canvasElement.height);
        
        // Convert canvas to data URL
        const imageDataURL = canvasElement.toDataURL('image/jpeg');
        
        // Show retry button
        captureBtn.style.display = 'none';
        retryBtn.style.display = 'block';
        
        // Process the captured image
        processWebcamImage(imageDataURL);
    });
    
    // Retry capture
    retryBtn.addEventListener('click', () => {
        // Show capture button, hide retry button
        captureBtn.style.display = 'block';
        retryBtn.style.display = 'none';
        
        // Reset UI
        resetUI();
    });
    
    // Handle file input change
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            const fileName = fileInput.files[0].name;
            fileNameDisplay.textContent = fileName;
        } else {
            fileNameDisplay.textContent = '';
        }
    });
    
    // Handle form submission
    uploadForm.addEventListener('submit', (e) => {
        e.preventDefault();
        
        if (fileInput.files.length === 0) {
            showError('Please select an image file.');
            return;
        }
        
        const file = fileInput.files[0];
        processUploadedImage(file);
    });
    
    // Process webcam image
    function processWebcamImage(imageDataURL) {
        // Show loading
        showLoading();
        
        // Send image to server
        fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: imageDataURL })
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Failed to process image');
                });
            }
            return response.json();
        })
        .then(data => {
            // Hide loading
            hideLoading();
            
            // Display results
            displayResults(data);
        })
        .catch(error => {
            // Hide loading
            hideLoading();
            
            // Show error
            showError(error.message);
        });
    }
    
    // Process uploaded image
    function processUploadedImage(file) {
        // Show loading
        showLoading();
        
        // Create form data
        const formData = new FormData();
        formData.append('file', file);
        
        // Send image to server
        fetch('/api/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Failed to process image');
                });
            }
            return response.json();
        })
        .then(data => {
            // Hide loading
            hideLoading();
            
            // Display results
            displayResults(data);
        })
        .catch(error => {
            // Hide loading
            hideLoading();
            
            // Show error
            showError(error.message);
        });
    }
    
    // Display results
    function displayResults(data) {
        // Show results container
        resultsElement.style.display = 'block';
        
        // Display emotions
        displayEmotions(data.faces);
        
        // Display recommendations
        displayRecommendations(data.recommendations);
        
        // Scroll to results
        resultsElement.scrollIntoView({ behavior: 'smooth' });
    }
    
    // Display emotions
    function displayEmotions(faces) {
        // Clear previous emotions
        emotionsListElement.innerHTML = '';
        
        if (!faces || faces.length === 0) {
            emotionsListElement.innerHTML = '<p>No emotions detected.</p>';
            return;
        }
        
        // Get the first face's emotions
        const faceEmotions = faces[0].emotions;
        
        // Create emotion items
        const emotionEntries = Object.entries(faceEmotions);
        emotionEntries.sort((a, b) => b[1] - a[1]); // Sort by probability (descending)
        
        // Prepare data for chart
        const labels = [];
        const data = [];
        const backgroundColors = [
            '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40', '#C9CBCF'
        ];
        
        emotionEntries.forEach(([emotion, probability], index) => {
            // Add to chart data
            labels.push(emotion.charAt(0).toUpperCase() + emotion.slice(1));
            data.push(probability);
            
            // Create emotion item
            const emotionItem = document.createElement('div');
            emotionItem.className = 'emotion-item';
            emotionItem.innerHTML = `
                <span>${emotion.charAt(0).toUpperCase() + emotion.slice(1)}</span>
                <span>${Math.round(probability * 100)}%</span>
            `;
            emotionsListElement.appendChild(emotionItem);
        });
        
        // Create or update chart - FIX: Make sure the canvas exists and is ready
        const chartCanvas = document.getElementById('emotions-chart');
        if (chartCanvas) {
            // Destroy previous chart if it exists
            if (emotionsChart) {
                emotionsChart.destroy();
            }
            
            // Make sure we have a 2D context
            const ctx = chartCanvas.getContext('2d');
            if (ctx) {
                emotionsChart = new Chart(ctx, {
                    type: 'doughnut',
                    data: {
                        labels: labels,
                        datasets: [{
                            data: data,
                            backgroundColor: backgroundColors.slice(0, data.length),
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                position: 'right',
                                labels: {
                                    boxWidth: 12,
                                    font: {
                                        size: 10
                                    }
                                }
                            }
                        }
                    }
                });
            } else {
                console.error('Could not get 2D context from emotions-chart canvas');
                // Fallback to text-only display (we already have the text list)
            }
        } else {
            console.error('Could not find emotions-chart canvas element');
            // Fallback to text-only display (we already have the text list)
        }
    }
    
    // Display recommendations
    function displayRecommendations(recommendations) {
        // Clear previous recommendations
        recommendationsElement.innerHTML = '';
        
        if (!recommendations || recommendations.length === 0) {
            recommendationsElement.innerHTML = '<p>No recommendations available.</p>';
            return;
        }
        
        // Create movie cards
        recommendations.forEach(movie => {
            const movieCard = document.createElement('div');
            movieCard.className = 'movie-card';
            
            // Poster image
            let posterUrl = '/static/img/no-poster.jpg'; // Default image
            if (movie.poster_path) {
                posterUrl = `/api/poster/${movie.poster_path}`;
            }
            
            // Genre tags
            const genreTags = movie.genres.map(genre => 
                `<span class="genre-tag">${genre}</span>`
            ).join('');
            
            movieCard.innerHTML = `
                <img class="movie-poster" src="${posterUrl}" alt="${movie.title} poster">
                <div class="movie-info">
                    <h3 class="movie-title">${movie.title}</h3>
                    <div class="movie-meta">
                        <span>${movie.year}</span>
                        <span class="movie-rating">★ ${movie.rating}</span>
                    </div>
                    <div class="movie-genres">
                        ${genreTags}
                    </div>
                    <p class="movie-overview">${movie.overview}</p>
                </div>
            `;
            
            recommendationsElement.appendChild(movieCard);
        });
    }
    
    // Show loading
    function showLoading() {
        loadingElement.style.display = 'block';
        resultsElement.style.display = 'none';
        errorMessageElement.style.display = 'none';
    }
    
    // Hide loading
    function hideLoading() {
        loadingElement.style.display = 'none';
    }
    
    // Show error
    function showError(message) {
        errorMessageElement.style.display = 'block';
        errorDetailsElement.textContent = message;
    }
    
    // Reset UI
    function resetUI() {
        resultsElement.style.display = 'none';
        errorMessageElement.style.display = 'none';
        loadingElement.style.display = 'none';
        fileNameDisplay.textContent = '';
        fileInput.value = '';
    }
    
    // Initialize webcam on page load if webcam tab is active
    if (document.querySelector('.tab-btn[data-tab="webcam"]').classList.contains('active')) {
        startWebcam();
    }
});