document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const fileInput = document.getElementById('fileInput');
    const uploadButton = document.getElementById('uploadButton');
    const uploadArea = document.getElementById('uploadArea');
    const resultsSection = document.getElementById('resultsSection');
    const previewImage = document.getElementById('previewImage');
    const weatherResult = document.getElementById('weatherResult');
    const timeResult = document.getElementById('timeResult');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const errorMessage = document.getElementById('errorMessage');
    const errorText = document.getElementById('errorText');

    // Weather icons mapping
    const weatherIcons = {
        'clear': 'fas fa-sun',
        'partly cloudy': 'fas fa-cloud-sun',
        'overcast': 'fas fa-cloud',
        'rainy': 'fas fa-cloud-rain',
        'snowy': 'fas fa-snowflake',
        'foggy': 'fas fa-smog'
    };

    // Time icons mapping
    const timeIcons = {
        'daytime': 'fas fa-sun',
        'dawn/dusk': 'fas fa-cloud-sun',
        'night': 'fas fa-moon'
    };

    // Handle file selection
    fileInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            uploadFile(e.target.files[0]);
        }
    });

    // Handle upload button click
    uploadButton.addEventListener('click', function() {
        fileInput.click();
    });

    // Handle drag and drop
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.style.borderColor = '#3498db';
        uploadArea.style.backgroundColor = '#f0f5ff';
    });

    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadArea.style.borderColor = '#bdc3c7';
        uploadArea.style.backgroundColor = 'white';
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.style.borderColor = '#bdc3c7';
        uploadArea.style.backgroundColor = 'white';

        if (e.dataTransfer.files.length > 0) {
            uploadFile(e.dataTransfer.files[0]);
        }
    });

    // Handle upload area click
    uploadArea.addEventListener('click', function() {
        fileInput.click();
    });

    function uploadFile(file) {
        // Validate file type
        const validTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/jpg'];
        if (!validTypes.includes(file.type)) {
            showError('Please upload a valid image file (JPG, PNG, GIF)');
            return;
        }

        // Validate file size (16MB max)
        if (file.size > 16 * 1024 * 1024) {
            showError('File size exceeds 16MB limit');
            return;
        }

        // Show loading overlay
        loadingOverlay.style.display = 'flex';

        // Create FormData and send to server
        const formData = new FormData();
        formData.append('file', file);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw err; });
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                // Display results
                previewImage.src = data.image_url;
                weatherResult.textContent = data.prediction.weather;
                timeResult.textContent = data.prediction.timeofday;

                // Update icons based on predictions
                updateWeatherIcon(data.prediction.weather);
                updateTimeIcon(data.prediction.timeofday);

                // Show results section
                resultsSection.style.display = 'block';
                uploadArea.style.display = 'none';
            } else {
                showError(data.error || 'Unknown error occurred');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showError(error.error || error.message || 'Failed to process image');
        })
        .finally(() => {
            loadingOverlay.style.display = 'none';
        });
    }

    function updateWeatherIcon(weather) {
        const iconElement = document.querySelector('.weather-result .result-icon i');
        if (iconElement) {
            iconElement.className = weatherIcons[weather.toLowerCase()] || 'fas fa-cloud-sun';
        }
    }

    function updateTimeIcon(time) {
        const iconElement = document.querySelector('.time-result .result-icon i');
        if (iconElement) {
            iconElement.className = timeIcons[time.toLowerCase()] || 'fas fa-clock';
        }
    }

    function showError(message) {
        errorText.textContent = message;
        errorMessage.style.display = 'block';

        // Hide error after 5 seconds
        setTimeout(() => {
            errorMessage.style.display = 'none';
        }, 5000);
    }

    // Add reset functionality
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && resultsSection.style.display === 'block') {
            // Reset the interface
            resultsSection.style.display = 'none';
            uploadArea.style.display = 'block';
            fileInput.value = '';
        }
    });

    // Handle "Upload Another Image" button
    const uploadAnotherButton = document.getElementById('uploadAnotherButton');
    if (uploadAnotherButton) {
        uploadAnotherButton.addEventListener('click', function() {
            // Reset the interface
            resultsSection.style.display = 'none';
            uploadArea.style.display = 'block';
            fileInput.value = '';
            previewImage.src = '';
        });
    }
});
