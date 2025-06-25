#!/usr/bin/env python3
"""
Startup script for the Spam Email Classifier web application.
This script checks system requirements and starts the Flask server.
"""

import os
import sys
import subprocess
import time
import webbrowser
from datetime import datetime

def print_banner():
    """Print startup banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                  🚀 SPAM EMAIL CLASSIFIER 🚀                 ║
║                     Web Application Startup                   ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

def check_models():
    """Check if trained models exist."""
    print("🔍 Checking trained models...")
    
    models_dir = "models"
    required_models = [
        "naive_bayes_model.joblib",
        "svm_model.joblib", 
        "random_forest_model.joblib",
        "logistic_regression_model.joblib"
    ]
    
    if not os.path.exists(models_dir):
        print(f"❌ Models directory '{models_dir}' not found!")
        return False
    
    missing_models = []
    for model in required_models:
        model_path = os.path.join(models_dir, model)
        if os.path.exists(model_path):
            print(f"   ✅ {model}")
        else:
            missing_models.append(model)
            print(f"   ❌ {model} - MISSING")
    
    if missing_models:
        print(f"\n⚠️  Missing {len(missing_models)} model(s)!")
        print("To train models, run: python3 train_models.py --no-dl")
        return False
    
    print("✅ All required models found!")
    return True

def check_dependencies():
    """Check if required Python packages are installed."""
    print("\n🔍 Checking dependencies...")
    
    required_packages = [
        "flask",
        "pandas", 
        "numpy",
        ("scikit-learn", "sklearn"),  # (package_name, import_name)
        "joblib"
    ]
    
    missing_packages = []
    for package in required_packages:
        if isinstance(package, tuple):
            package_name, import_name = package
        else:
            package_name = import_name = package
            
        try:
            __import__(import_name)
            print(f"   ✅ {package_name}")
        except ImportError:
            missing_packages.append(package_name)
            print(f"   ❌ {package_name} - MISSING")
    
    if missing_packages:
        print(f"\n⚠️  Missing {len(missing_packages)} package(s)!")
        print("To install missing packages, run: pip3 install -r requirements.txt")
        return False
    
    print("✅ All dependencies satisfied!")
    return True

def test_app_components():
    """Test if app components work."""
    print("\n🔍 Testing application components...")
    
    try:
        # Add src to path
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        
        # Test model loading
        from traditional_ml import TraditionalMLModels
        ml_models = TraditionalMLModels()
        ml_models.load_models('models')
        print(f"   ✅ Loaded {len(ml_models.trained_models)} ML models")
        
        # Test ensemble
        from deep_learning import EnsemblePredictor, DeepLearningModels
        dl_models = DeepLearningModels()
        ensemble = EnsemblePredictor(ml_models, dl_models)
        print("   ✅ Ensemble predictor created")
        
        # Test prediction
        result = ensemble.predict("Test email")
        print(f"   ✅ Prediction system working: {result['prediction']}")
        
        print("✅ All components working correctly!")
        return True
        
    except Exception as e:
        print(f"   ❌ Component test failed: {str(e)}")
        return False

def start_flask_app():
    """Start the Flask application."""
    print("\n🚀 Starting Flask application...")
    
    try:
        # Import and run the Flask app
        print("   Loading Flask application...")
        
        # Run the app with proper error handling
        env = os.environ.copy()
        env['PYTHONPATH'] = os.path.join(os.getcwd(), 'src')
        
        print("   🌐 Starting server (will auto-detect available port)")
        print("   📝 Server logs will appear below...")
        print("   💡 Press Ctrl+C to stop the server")
        print("\n" + "="*60)
        
        # Start the Flask app
        subprocess.run([sys.executable, "app.py"], env=env, check=True)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Flask app failed to start: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check if port 5000 is available")
        print("2. Verify all dependencies are installed")
        print("3. Check app.py for syntax errors")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

def open_browser():
    """Open browser to the application."""
    print("\n🌐 Opening web browser...")
    
    # Try common ports
    ports_to_try = [5000, 5001, 5002, 8000, 8080]
    
    for port in ports_to_try:
        url = f"http://localhost:{port}"
        try:
            # Wait a moment for server to start
            time.sleep(2)
            
            # Test if server is responding
            import urllib.request
            urllib.request.urlopen(url, timeout=2)
            
            # If we get here, server is responding
            webbrowser.open(url)
            print(f"   ✅ Browser opened to {url}")
            return
            
        except:
            continue
    
    # If no server found, just open localhost:5000 as default
    url = "http://localhost:5000"
    try:
        webbrowser.open(url)
        print(f"   ⚠️  Opened default URL: {url}")
        print(f"   📝 If server is on different port, check terminal output")
    except Exception as e:
        print(f"   ⚠️  Could not auto-open browser: {e}")
        print(f"   📝 Please manually open the URL shown in terminal output")

def main():
    """Main startup function."""
    print_banner()
    print(f"🕒 Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check prerequisites
    if not check_models():
        print("\n❌ Startup failed: Missing trained models")
        print("\n💡 Quick fix:")
        print("   python3 train_models.py --no-dl")
        return False
    
    if not check_dependencies():
        print("\n❌ Startup failed: Missing dependencies")
        print("\n💡 Quick fix:")
        print("   pip3 install -r requirements.txt")
        return False
    
    if not test_app_components():
        print("\n❌ Startup failed: Component test failed")
        return False
    
    print("\n🎉 All checks passed! Ready to start web application.")
    
    # Ask user if they want to auto-open browser
    try:
        response = input("\n🌐 Open browser automatically? (y/n): ").strip().lower()
        auto_open = response in ['y', 'yes', '']
    except KeyboardInterrupt:
        print("\n🛑 Startup cancelled by user")
        return False
    
    # Start the Flask app
    if auto_open:
        # Start server in background and open browser
        import threading
        server_thread = threading.Thread(target=start_flask_app)
        server_thread.daemon = True
        server_thread.start()
        
        open_browser()
        
        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n🛑 Application stopped")
    else:
        start_flask_app()
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Startup interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
