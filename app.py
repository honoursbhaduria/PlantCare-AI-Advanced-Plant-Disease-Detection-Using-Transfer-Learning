"""
PlantCare AI - Advanced Plant Disease Detection Using Transfer Learning
Flask Backend Application
"""

import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MODEL_PATH = 'model/plantcare_model.h5'
INPUT_SHAPE = (224, 224)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

# Class labels
CLASS_NAMES = ['Bacterial Leaf Blight', 'Brown Spot', 'Leaf Smut']

# Disease information
DISEASE_INFO = {
    'Bacterial Leaf Blight': {
        'cause': 'Xanthomonas oryzae pv. oryzae (bacterium)',
        'symptoms': 'Water-soaked lesions on leaves that turn brown and dry. In severe cases, wilting and death of the plant.',
        'treatment': 'Use resistant varieties, apply copper-based bactericides, ensure proper field drainage, and practice crop rotation.',
        'severity': 'High',
        'color': '#dc3545'
    },
    'Brown Spot': {
        'cause': 'Cochliobolus miyabeanus (fungus)',
        'symptoms': 'Small oval to elliptical brown spots with yellow halo on leaves. Spots can coalesce causing leaves to wither.',
        'treatment': 'Use certified disease-free seeds, apply fungicides (e.g., Mancozeb), maintain balanced fertilization, and ensure proper soil nutrition.',
        'severity': 'Medium',
        'color': '#fd7e14'
    },
    'Leaf Smut': {
        'cause': 'Entyloma oryzae (fungus)',
        'symptoms': 'Small, round, reddish-brown spots on leaves that turn black and produce powdery spores.',
        'treatment': 'Remove infected plant debris, apply appropriate fungicides, and use resistant cultivars where available.',
        'severity': 'Low',
        'color': '#ffc107'
    }
}

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = None
def load_trained_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}. Predictions will use demo mode.")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_disease(img_path):
    """Predict disease from an image file."""
    if model is None:
        # Demo mode - return a random prediction for testing
        idx = np.random.randint(0, len(CLASS_NAMES))
        confidence = round(np.random.uniform(0.85, 0.99), 4)
        return CLASS_NAMES[idx], confidence, {name: round(np.random.uniform(0.01, 0.15), 4) if i != idx else confidence for i, name in enumerate(CLASS_NAMES)}

    img = image.load_img(img_path, target_size=INPUT_SHAPE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    predicted_idx = np.argmax(predictions)
    confidence = float(predictions[predicted_idx])
    all_predictions = {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}

    return CLASS_NAMES[predicted_idx], confidence, all_predictions


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file selected. Please upload an image.')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No file selected. Please upload an image.')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        predicted_class, confidence, all_predictions = predict_disease(filepath)
        disease_details = DISEASE_INFO.get(predicted_class, {})

        return render_template('result.html',
                               prediction=predicted_class,
                               confidence=round(confidence * 100, 2),
                               all_predictions={k: round(v * 100, 2) for k, v in all_predictions.items()},
                               disease_info=disease_details,
                               image_path=filepath)
    else:
        return render_template('index.html', error='Invalid file type. Please upload a JPG, PNG, or WEBP image.')


@app.route('/about')
def about():
    return render_template('about.html')


# Load model at module level for production (Render/Vercel)
load_trained_model()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
