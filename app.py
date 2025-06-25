from flask import Flask, render_template, request, jsonify
import os
import sys
import joblib
import torch
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from traditional_ml import TraditionalMLModels
from deep_learning import DeepLearningModels, EnsemblePredictor

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')

class SpamClassifierApp:
    """Flask application for spam email classification."""
    
    def __init__(self):
        self.ml_models = TraditionalMLModels()
        self.dl_models = DeepLearningModels()
        self.ensemble = None
        self.load_models()
    
    def load_models(self):
        """Load trained models."""
        print("Loading models...")
        
        # Load traditional ML models
        try:
            self.ml_models.load_models('models')
            print(f"Loaded {len(self.ml_models.trained_models)} traditional ML models")
        except Exception as e:
            print(f"Error loading ML models: {str(e)}")
        
        # Load deep learning models
        try:
            model_dir = 'models'
            for model_name in ['bert-base', 'distilbert']:
                model_path = os.path.join(model_dir, model_name)
                if os.path.exists(model_path):
                    self.dl_models.load_model(model_name, model_path)
            
            print(f"Loaded {len(self.dl_models.trained_models)} deep learning models")
        except Exception as e:
            print(f"Error loading DL models: {str(e)}")
        
        # Create ensemble predictor
        if self.ml_models.trained_models or self.dl_models.trained_models:
            self.ensemble = EnsemblePredictor(self.ml_models, self.dl_models)
            print("Ensemble predictor created")
        else:
            print("No models loaded - predictions will not be available")

# Initialize the classifier
classifier_app = SpamClassifierApp()

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make spam prediction."""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if classifier_app.ensemble is None:
            return jsonify({'error': 'No models loaded'}), 500
        
        # Make prediction
        result = classifier_app.ensemble.predict(text)
        
        # Add timestamp
        result['timestamp'] = datetime.now().isoformat()
        result['text_length'] = len(text)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_individual', methods=['POST'])
def predict_individual():
    """Get predictions from individual models."""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        results = {}
        
        # Get ML predictions
        if classifier_app.ml_models.trained_models:
            ml_predictions = classifier_app.ml_models.predict(text)
            results['traditional_ml'] = ml_predictions
        
        # Get DL predictions
        if classifier_app.dl_models.trained_models:
            dl_predictions = classifier_app.dl_models.predict(text)
            results['deep_learning'] = dl_predictions
        
        if not results:
            return jsonify({'error': 'No models available'}), 500
        
        results['timestamp'] = datetime.now().isoformat()
        results['text_length'] = len(text)
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/models_info')
def models_info():
    """Get information about loaded models."""
    try:
        info = {
            'traditional_ml_models': list(classifier_app.ml_models.trained_models.keys()),
            'deep_learning_models': list(classifier_app.dl_models.trained_models.keys()),
            'total_models': len(classifier_app.ml_models.trained_models) + len(classifier_app.dl_models.trained_models),
            'ensemble_available': classifier_app.ensemble is not None
        }
        
        return jsonify(info)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': classifier_app.ensemble is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Check if we're running in production (Heroku sets PORT env var)
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    
    if os.environ.get('PORT'):
        # Production environment (Heroku)
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        # Development environment
        import socket
        
        # Try different ports if 5000 is in use
        ports_to_try = [5000, 5001, 5002, 8000, 8080]
        
        for port in ports_to_try:
            try:
                # Test if port is available
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                
                if result != 0:  # Port is available
                    print(f"Starting server on port {port}")
                    app.run(debug=True, host='0.0.0.0', port=port)
                    break
                else:
                    print(f"Port {port} is in use, trying next...")
                    
            except Exception as e:
                print(f"Error testing port {port}: {e}")
                continue
        else:
            print("All ports are in use. Please free up a port or specify a different one.")
            print("On macOS, you may need to disable AirPlay Receiver in System Preferences.")
            exit(1)
