#!/usr/bin/env python3
"""
Simple test script to verify web application functionality.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_web_app_components():
    """Test that all web app components are working."""
    print("Testing Web Application Components")
    print("=" * 50)
    
    try:
        # Test model loading
        print("1. Testing model loading...")
        from traditional_ml import TraditionalMLModels
        ml_models = TraditionalMLModels()
        ml_models.load_models('models')
        print(f"   ✓ Loaded {len(ml_models.trained_models)} models")
        
        # Test prediction
        print("2. Testing prediction functionality...")
        test_emails = [
            "FREE MONEY! Click here now!",
            "Hey, how are you doing today?"
        ]
        
        for email in test_emails:
            predictions = ml_models.predict(email)
            print(f"   ✓ '{email[:30]}...' -> {len(predictions)} predictions")
        
        # Test app imports
        print("3. Testing Flask app imports...")
        from deep_learning import DeepLearningModels, EnsemblePredictor
        print("   ✓ Deep learning modules imported")
        
        from flask import Flask
        print("   ✓ Flask imported")
        
        print("\n🎉 All components working correctly!")
        print("\nTo start the web application:")
        print("   python3 app.py")
        print("   Then open http://localhost:5000 in your browser")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_web_app_components()
