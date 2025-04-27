from flask import Flask, request, jsonify
import os
import sys

# Add 3detr directory to sys.path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '3detr'))
from detector import run_detection

app = Flask(__name__)

@app.route('/infer', methods=['POST'])
def infer():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    # Save the uploaded file
    filepath = os.path.join('/tmp', file.filename)
    file.save(filepath)
    # Run the 3DETR model on the uploaded file
    try:
        result = run_detection(filepath)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 