from flask import Flask, request, jsonify, render_template
import os
import uuid
import subprocess
import json
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

            # Run inference using the existing script
            result = subprocess.run([
                'python', '-m', 'inference.infer',
                '--image', filepath,
                '--model', 'checkpoints/best_model.pth'
            ], capture_output=True, text=True, cwd='.')

            if result.returncode != 0:
                return jsonify({
                    'error': 'Inference failed',
                    'details': result.stderr
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
