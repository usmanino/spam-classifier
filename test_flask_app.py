#!/usr/bin/env python3
"""
Simple test Flask app to check if everything works locally.
"""

from flask import Flask, render_template, request, jsonify
import os
import sys
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from traditional_ml import TraditionalMLModels
from deep_learning import DeepLearningModels, EnsemblePredictor

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')

# Initialize models
print("Initializing models...")
ml_models = TraditionalMLModels()
dl_models = DeepLearningModels()

try:
    ml_models.load_models('models')
    print(f"✓ Loaded {len(ml_models.trained_models)} ML models")
except Exception as e:
    print(f"✗ Error loading ML models: {e}")

try:
    ensemble = EnsemblePredictor(ml_models, dl_models)
    print("✓ Ensemble predictor created")
except Exception as e:
    print(f"✗ Error creating ensemble: {e}")
    ensemble = None

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make spam prediction."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400
            
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if ensemble is None:
            return jsonify({'error': 'Ensemble predictor not available'}), 500
        
        # Make prediction
        result = ensemble.predict(text)
        
        # Add metadata
        result['timestamp'] = datetime.now().isoformat()
        result['text_length'] = len(text)
        
        print(f"Prediction for '{text[:50]}...': {result['prediction']} ({result['confidence']:.3f})")
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in /predict: {str(e)}")
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/predict_individual', methods=['POST'])
def predict_individual():
    """Get predictions from individual models."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400
            
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        results = {}
        
        # Get ML predictions
        if ml_models.trained_models:
            ml_predictions = ml_models.predict(text)
            results['traditional_ml'] = ml_predictions
            print(f"ML predictions: {len(ml_predictions)} models")
        
        # Get DL predictions (none available yet)
        if dl_models.trained_models:
            dl_predictions = dl_models.predict(text)
            results['deep_learning'] = dl_predictions
            print(f"DL predictions: {len(dl_predictions)} models")
        
        if not results:
            return jsonify({'error': 'No models available'}), 500
        
        results['timestamp'] = datetime.now().isoformat()
        results['text_length'] = len(text)
        
        return jsonify(results)
    
    except Exception as e:
        print(f"Error in /predict_individual: {str(e)}")
        return jsonify({'error': f'Individual prediction error: {str(e)}'}), 500

@app.route('/models_info')
def models_info():
    """Get information about loaded models."""
    try:
        info = {
            'traditional_ml_models': list(ml_models.trained_models.keys()),
            'deep_learning_models': list(dl_models.trained_models.keys()),
            'total_models': len(ml_models.trained_models) + len(dl_models.trained_models),
            'ensemble_available': ensemble is not None
        }
        
        return jsonify(info)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': ensemble is not None,
        'ml_models': len(ml_models.trained_models),
        'dl_models': len(dl_models.trained_models),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/test')
def test():
    """Simple test endpoint."""
    return jsonify({
        'message': 'Flask app is working!',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Starting Simple Test Flask App")
    print("="*50)
    print(f"ML Models: {len(ml_models.trained_models)}")
    print(f"DL Models: {len(dl_models.trained_models)}")
    print(f"Ensemble: {'Available' if ensemble else 'Not Available'}")
    print("="*50)
    print("Open http://localhost:5000 in your browser")
    print("="*50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
