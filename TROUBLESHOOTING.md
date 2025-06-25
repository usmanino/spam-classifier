# 🚨 Troubleshooting Guide

## Common Issues and Solutions

### 1. "An error occurred while classifying the email" in Web Interface

**Problem**: The web interface shows this error when trying to classify emails.

**Cause**: The Flask server is not running or not reachable.

**Solutions**:

#### ✅ **Quick Fix**:
```bash
# Make sure you're in the project directory
cd /Users/factorial/Programming/spam-classifier

# Start the Flask server
python3 app.py
```

#### ✅ **Use the Startup Script**:
```bash
python3 start_app.py
```

#### ✅ **Manual Verification**:
1. Open terminal
2. Navigate to project directory: `cd /Users/factorial/Programming/spam-classifier`
3. Check if models exist: `ls models/`
4. Should see: `naive_bayes_model.joblib`, `svm_model.joblib`, etc.
5. Start server: `python3 app.py`
6. Look for: `Running on http://127.0.0.1:5000`
7. Open browser to: `http://localhost:5000`

---

### 2. "No trained models found" Error

**Problem**: Models are missing from the `models/` directory.

**Solutions**:

#### ✅ **Train Models**:
```bash
# Train traditional ML models (faster)
python3 train_models.py --no-dl

# Or train all models (includes deep learning)
python3 train_models.py
```

#### ✅ **Check Models Directory**:
```bash
ls -la models/
```
Should contain:
- `naive_bayes_model.joblib`
- `svm_model.joblib`
- `random_forest_model.joblib` 
- `logistic_regression_model.joblib`

---

### 3. Import Errors or Missing Dependencies

**Problem**: Python packages are missing.

**Solutions**:

#### ✅ **Install Dependencies**:
```bash
pip3 install -r requirements.txt
```

#### ✅ **Check Python Version**:
```bash
python3 --version
```
Should be Python 3.7 or higher.

#### ✅ **Alternative Installation**:
```bash
# If pip3 doesn't work, try:
python3 -m pip install -r requirements.txt

# Or with user flag:
pip3 install --user -r requirements.txt
```

---

### 4. NLTK Data Download Issues

**Problem**: NLTK data download timeout errors.

**Solutions**:

#### ✅ **Offline Operation**:
The system now works without NLTK data downloads. If you see NLTK errors, they can be safely ignored as fallback mechanisms are in place.

#### ✅ **Manual NLTK Download** (Optional):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

### 5. Port 5000 Already in Use

**Problem**: Flask can't start because port 5000 is occupied.

**Solutions**:

#### ✅ **Find and Kill Process**:
```bash
# Find process using port 5000
lsof -ti:5000

# Kill the process (replace PID with actual process ID)
kill -9 PID
```

#### ✅ **Use Different Port**:
```bash
# Modify app.py to use port 5001
# Change: app.run(debug=True, host='0.0.0.0', port=5000)
# To: app.run(debug=True, host='0.0.0.0', port=5001)
```

---

### 6. Web Interface Not Loading

**Problem**: Browser shows "This site can't be reached" or similar.

**Solutions**:

#### ✅ **Check Server Status**:
1. Look for "Running on http://127.0.0.1:5000" in terminal
2. Try different URL formats:
   - `http://localhost:5000`
   - `http://127.0.0.1:5000`
   - `http://0.0.0.0:5000`

#### ✅ **Test Basic Connectivity**:
```bash
# Test if server is responding
curl http://localhost:5000/health
```

---

### 7. Poor Model Performance

**Problem**: Models always predict the same class or show low accuracy.

**Solutions**:

#### ✅ **Retrain with More Data**:
1. Add more diverse email samples to `src/data_preprocessing.py`
2. Retrain: `python3 train_models.py --no-dl`

#### ✅ **Evaluate Models**:
```bash
python3 evaluate_models.py
```

---

## 🔧 Debug Commands

### System Status Check
```bash
python3 system_report.py
```

### Test Individual Components
```bash
python3 test_web_app.py
```

### Run Interactive Demo
```bash
python3 demo.py
```

### Quick Model Test
```bash
python3 -c "
import sys
sys.path.append('src')
from traditional_ml import TraditionalMLModels
models = TraditionalMLModels()
models.load_models('models')
result = models.predict('FREE MONEY! Click now!')
print('Test prediction:', result)
"
```

---

## 📞 Getting Help

If none of these solutions work:

1. **Check System Report**: `python3 system_report.py`
2. **Run Debug Test**: `python3 test_web_app.py`
3. **Review Logs**: Look for error messages in terminal output
4. **Check File Permissions**: Ensure you have read/write access to project directory

## 🎯 Quick Start Checklist

- [ ] Project directory: `/Users/factorial/Programming/spam-classifier`
- [ ] Dependencies installed: `pip3 install -r requirements.txt`
- [ ] Models trained: `python3 train_models.py --no-dl`
- [ ] Server started: `python3 app.py` or `python3 start_app.py`
- [ ] Browser opened to: `http://localhost:5000`

**✅ Everything working? You should see the spam classifier web interface!**
