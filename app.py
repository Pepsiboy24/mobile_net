#!/usr/bin/env python3
"""
TrustTag PWA Flask Server
Provides API endpoints for counterfeit product detection using trained ML model.
"""

import os
import pickle
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables for model components
model = None
label_encoder = None
feature_params = None

def load_models():
    """Load the trained model and preprocessing components"""
    global model, label_encoder, feature_params
    
    try:
        # Load the model
        with open('models/trusttag_v1.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load the label encoder
        with open('models/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Load feature parameters
        with open('models/feature_params.pkl', 'rb') as f:
            feature_params = pickle.load(f)
        
        logger.info("✅ Model and components loaded successfully")
        logger.info(f"Model type: {type(model).__name__}")
        logger.info(f"Classes: {list(label_encoder.classes_)}")
        logger.info(f"Feature parameters: {feature_params}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return False

def extract_histogram_texture_features(images, img_size):
    """
    Extract the same features used during training (color histograms + texture)
    """
    features = []
    
    for img_flat in images:
        # Reshape back to image shape
        img = img_flat.reshape(img_size[0], img_size[1], 3)
        
        # Extract color histogram features (16 bins per channel)
        r_hist = np.histogram(img[:,:,0], bins=16, range=(0,1))[0]
        g_hist = np.histogram(img[:,:,1], bins=16, range=(0,1))[0]
        b_hist = np.histogram(img[:,:,2], bins=16, range=(0,1))[0]
        
        # Extract texture features (simple gradients)
        gray = np.mean(img, axis=2)
        grad_x = np.abs(np.diff(gray, axis=0))
        grad_y = np.abs(np.diff(gray, axis=1))
        texture_feat = [np.mean(grad_x), np.mean(grad_y), np.std(grad_x), np.std(grad_y)]
        
        # Combine features (48 histogram features + 4 texture features = 52 total)
        feature_vector = np.concatenate([r_hist, g_hist, b_hist, texture_feat])
        features.append(feature_vector)
    
    return np.array(features)

def preprocess_image(image_file, feature_params):
    """Preprocess uploaded image for prediction using the same feature extraction as training"""
    try:
        # Load and resize image to the size used in training
        img_size = feature_params['img_size']
        img = Image.open(image_file).convert('RGB').resize(img_size)
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        # Flatten for feature extraction
        img_flat = img_array.flatten()
        
        # Extract features using the same method as training
        if feature_params['feature_type'] == 'histogram_texture':
            features = extract_histogram_texture_features([img_flat], img_size)[0]
        else:
            # Fallback to simple flattening if feature type is unknown
            features = img_flat
        
        return features, img
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

def predict_product(features):
    """Make prediction using the loaded model"""
    try:
        # Make prediction
        prediction = model.predict([features])[0]
        predicted_label = label_encoder.inverse_transform([prediction])[0]
        
        # Get prediction probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba([features])[0]
            confidence = float(max(probabilities))
            
            # Get probabilities for all classes
            class_probabilities = {}
            for i, class_name in enumerate(label_encoder.classes_):
                class_probabilities[class_name] = float(probabilities[i])
        else:
            confidence = 1.0
            class_probabilities = {predicted_label: 1.0}
        
        return {
            'predicted_class': predicted_label,
            'confidence': confidence,
            'all_probabilities': class_probabilities
        }
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise

@app.route('/')
def index():
    """Serve the main PWA interface"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for image prediction"""
    try:
        # Check if image file was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Validate image file
        if not image_file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            return jsonify({'error': 'Invalid image format. Supported formats: PNG, JPG, JPEG, WEBP'}), 400
        
        # Preprocess image and extract features
        features, display_image = preprocess_image(image_file, feature_params)
        
        # Make prediction
        result = predict_product(features)
        
        # Add additional info
        result.update({
            'status': 'success',
            'model_type': type(model).__name__,
            'feature_count': len(features)
        })
        
        logger.info(f"Prediction: {result['predicted_class']} with {result['confidence']:.2%} confidence")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'classes': list(label_encoder.classes_) if label_encoder else None
    })

@app.route('/sw.js')
def service_worker():
    """Serve the service worker file"""
    return app.send_static_file('sw.js')

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle not found error"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error"""
    return jsonify({'error': 'Internal server error'}), 500

def main():
    """Main function to start the Flask server"""
    logger.info("🚀 Starting TrustTag PWA Server...")
    
    # Load models before starting the server
    if not load_models():
        logger.error("❌ Failed to load models. Server cannot start.")
        return
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Configure Flask app
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['SECRET_KEY'] = 'trusttag-secret-key-2024'
    
    # Create uploads directory
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    logger.info("🌐 Starting Flask server on http://localhost:5000")
    logger.info("📱 TrustTag PWA is ready to use!")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    main()
