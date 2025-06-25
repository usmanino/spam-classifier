#!/bin/bash

# 🚀 Quick Local Test Script
echo "🧪 Testing Spam Email Classifier locally..."

# Check if models exist
if [ -d "models" ] && [ "$(ls -A models)" ]; then
    echo "✅ Models directory found with files"
    ls -la models/
else
    echo "⚠️  Models directory is empty or missing"
    echo "📝 You may need to train models first with: python train_models.py"
fi

# Check requirements
if [ -f "requirements.txt" ]; then
    echo "✅ Requirements file found"
else
    echo "❌ requirements.txt missing"
fi

# Check main app file
if [ -f "app.py" ]; then
    echo "✅ Main app.py found"
else
    echo "❌ app.py missing"
fi

# Check if virtual environment is recommended
echo ""
echo "🔧 Recommended next steps:"
echo "1. Create virtual environment: python3 -m venv venv"
echo "2. Activate it: source venv/bin/activate"
echo "3. Install dependencies: pip install -r requirements.txt"
echo "4. Test locally: python app.py"
echo "5. Deploy to your chosen platform!"

echo ""
echo "🌐 Ready for deployment platforms:"
echo "✅ Railway (railway.app) - Recommended"
echo "✅ Heroku (heroku.com)" 
echo "✅ Render (render.com)"
echo "✅ Docker ready"
