from flask import Flask, request, jsonify, render_template
import os
import uuid
import subprocess
import json
import sys  # Add this import
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = str(uuid.uuid4()) + secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            # Save the file
            file.save(filepath)
            
            print(f"File saved to: {filepath}")  # Debug log

            # Run inference using the existing script
            # Update these paths to match your trained model
            backbone = 'efficientnet_b0'  # Change this to match your model
            model_path = f'checkpoints/{backbone}/best_model_{backbone}.pth'
            
            # Check if model file exists
            if not os.path.exists(model_path):
                print(f"ERROR: Model file not found: {model_path}")
                # Fallback to old checkpoint if it exists
                if os.path.exists('checkpoints/best_model.pth'):
                    print("Using fallback: checkpoints/best_model.pth")
                    model_path = 'checkpoints/best_model.pth'
                    backbone = 'resnet18'
                else:
                    return jsonify({
                        'error': 'Model checkpoint not found',
                        'details': f'Looking for: {model_path}'
                    }), 500
            
            print(f"Using model: {model_path}, backbone: {backbone}")  # Debug log
            print(f"Python executable: {sys.executable}")  # Debug log
            
            result = subprocess.run([
                sys.executable, '-m', 'inference.infer',  # Use same Python as Flask
                '--image', filepath,
                '--model', model_path,
                '--backbone', backbone
            ], capture_output=True, text=True, cwd='.')
            
            print(f"Inference return code: {result.returncode}")  # Debug log
            print(f"Inference stdout: {result.stdout}")  # Debug log
            print(f"Inference stderr: {result.stderr}")  # Debug log

            if result.returncode != 0:
                return jsonify({
                    'error': 'Inference failed',
                    'details': result.stderr,
                    'stdout': result.stdout
                }), 500

            # Parse the JSON output
            try:
                prediction = json.loads(result.stdout)
                return jsonify({
                    'success': True,
                    'image_url': f'/static/uploads/{filename}',
                    'prediction': prediction
                })
            except json.JSONDecodeError:
                return jsonify({
                    'error': 'Failed to parse inference result',
                    'details': result.stdout
                }), 500

        except Exception as e:
            return jsonify({'error': str(e)}), 500
        # Note: We don't delete the file immediately so it can be displayed
        # Files will be cleaned up periodically or on server restart

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
