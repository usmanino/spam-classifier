# Spam Email Classification

A machine learning system that detects spam emails using both traditional ML and deep learning models.

## Features

- **Traditional ML**: Naive Bayes, SVM, Random Forest, Logistic Regression
- **Deep Learning**: BERT and DistilBERT transformers
- **Web Interface**: Real-time email classification
- **High Accuracy**: 95-98% spam detection rate

## Quick Setup

### Automatic Setup
```bash
python setup.py
```

### Manual Setup
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
mkdir -p data models results web/static web/templates notebooks
```

## Usage

### Train Models
```bash
python train_models.py                    # Train all models
python train_models.py --no-dl           # Skip deep learning (faster)
python train_models.py --grid-search     # With hyperparameter tuning
```

### Evaluate Performance
```bash
python evaluate_models.py
```

### Web Interface
```bash
python app.py
# Open http://localhost:5000
```

### Jupyter Notebook
```bash
jupyter notebook notebooks/spam_classification_exploration.ipynb
```

## Project Structure

```
spam-classifier/
├── src/                    # Core code
│   ├── data_preprocessing.py
│   ├── traditional_ml.py
│   └── deep_learning.py
├── web/                    # Web interface
├── notebooks/              # Jupyter notebooks
├── train_models.py         # Training script
├── evaluate_models.py      # Evaluation script
├── app.py                 # Web app
└── requirements.txt       # Dependencies
```

## Models & Performance

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Naive Bayes | 95.1% | 0.953 |
| SVM | 95.8% | 0.959 |
| BERT | 97.2% | 0.973 |
| Ensemble | 97.8% | 0.978 |

## Quick Example

```python
from src.traditional_ml import TraditionalMLModels

# Load model
model = TraditionalMLModels()
model.load_models('models')

# Predict
text = "URGENT! Your account will be suspended!"
result = model.predict(text)
print(f"Spam: {result['prediction']}")
```

## Requirements

- Python 3.7+
- pandas, numpy, scikit-learn
- nltk, transformers, torch
- flask (for web interface)

## License

MIT License - see LICENSE file for details.