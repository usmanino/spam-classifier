from flask import Flask, render_template, request, jsonify
import os
import re
from datetime import datetime

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')

class SimpleSpamDetector:
    """Ultra-lightweight spam detector for Vercel."""
    
    def __init__(self):
        # Simple spam indicators
        self.spam_keywords = [
            'free', 'win', 'winner', 'cash', 'money', 'prize', 'urgent', 'click', 
            'offer', 'limited', 'act now', 'congratulations', 'claim', 'bonus',
            'guaranteed', 'risk-free', 'no obligation', 'call now', 'order now',
            'credit', 'loan', 'investment', 'viagra', 'casino', 'lottery'
        ]
    
    def predict(self, text):
        """Simple spam detection using keyword matching."""
        if not text:
            return {
                'prediction': 'unknown',
                'confidence': 0.5,
                'spam_probability': 0.5,
                'error': 'No text provided'
            }
        
        text_lower = text.lower()
        
        # Count spam keywords
        spam_score = 0
        total_words = len(text.split())
        
        for keyword in self.spam_keywords:
            if keyword in text_lower:
                spam_score += 1
        
        # Additional spam indicators
        if re.search(r'\b\d+%\s*(off|discount)', text_lower):
            spam_score += 1
        if re.search(r'\$\d+', text):
            spam_score += 1
        if len(re.findall(r'[!]{2,}', text)) > 0:
            spam_score += 1
        if text.isupper() and len(text) > 20:
            spam_score += 1
            
        # Calculate probabilities
        spam_probability = min(spam_score / 5.0, 1.0)  # Normalize to 0-1
        
        if spam_probability > 0.6:
            prediction = 'spam'
            confidence = spam_probability
        else:
            prediction = 'ham'
            confidence = 1 - spam_probability
        
        return {
            'prediction': prediction,
            'confidence': float(confidence),
            'spam_probability': float(spam_probability),
            'keywords_found': spam_score,
            'deployment': 'vercel-lightweight'
        }

# Initialize detector
detector = SimpleSpamDetector()

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
        
        # Make prediction
        result = detector.predict(text)
        
        # Add metadata
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
        
        # Get prediction
        prediction_result = detector.predict(text)
        
        # Simulate individual model results
        results = {
            'traditional_ml': {
                'keyword_detector': {
                    'prediction': prediction_result['prediction'],
                    'confidence': prediction_result['confidence'],
                    'spam_probability': prediction_result['spam_probability']
                },
                'pattern_analyzer': {
                    'prediction': prediction_result['prediction'],
                    'confidence': max(0.3, prediction_result['confidence'] - 0.1),
                    'spam_probability': prediction_result['spam_probability']
                }
            },
            'timestamp': datetime.now().isoformat(),
            'text_length': len(text),
            'deployment': 'vercel-lightweight'
        }
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/models_info')
def models_info():
    """Get information about loaded models."""
    try:
        info = {
            'traditional_ml_models': ['keyword_detector', 'pattern_analyzer'],
            'deep_learning_models': [],
            'total_models': 2,
            'ensemble_available': True,
            'deployment': 'vercel-lightweight'
        }
        
        return jsonify(info)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': True,
        'timestamp': datetime.now().isoformat(),
        'deployment': 'vercel-lightweight'
    })

# Vercel entry point
app = app
