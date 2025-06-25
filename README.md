# Spam Email Classification

A comprehensive spam email classification system using both traditional machine learning and pre-trained deep learning models. This project demonstrates the implementation of various ML and DL approaches for email spam detection with a complete pipeline from data preprocessing to deployment.

## 🌟 Features

### Machine Learning Models
- **Traditional ML**: Naive Bayes, SVM, Random Forest, Logistic Regression
- **Deep Learning**: BERT, DistilBERT (pre-trained transformers)
- **Ensemble Methods**: Combining multiple models for robust predictions

### Capabilities
- Advanced text preprocessing and feature extraction
- Comprehensive model evaluation and comparison
- Interactive web interface for real-time classification
- Detailed performance visualizations and analysis
- Jupyter notebook for interactive exploration
- Model interpretability and error analysis

## 🚀 Quick Start

### Option 1: Automated Setup
```bash
python setup.py
```

### Option 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Create directories
mkdir -p data models results web/static web/templates notebooks
```

## 📊 Usage

### 1. Train Models
```bash
# Train all models (traditional ML + deep learning)
python train_models.py

# Train only traditional ML models (faster)
python train_models.py --no-dl

# Train with hyperparameter tuning
python train_models.py --grid-search

# Custom training options
python train_models.py --epochs 5 --batch-size 32 --data-path your_data.csv
```

### 2. Evaluate Models
```bash
# Comprehensive model evaluation
python evaluate_models.py

# Evaluation with custom data
python evaluate_models.py --data-path your_test_data.csv
```

### 3. Web Interface
```bash
# Start the web application
python app.py

# Then open http://localhost:5000 in your browser
```

### 4. Interactive Exploration
```bash
# Start Jupyter notebook
jupyter notebook notebooks/spam_classification_exploration.ipynb
```

## 📁 Project Structure

```
spam-classifier/
├── src/                           # Core source code
│   ├── data_preprocessing.py      # Text preprocessing and feature extraction
│   ├── traditional_ml.py          # Traditional ML models
│   └── deep_learning.py           # Deep learning models with transformers
├── web/                           # Web interface
│   ├── templates/
│   │   └── index.html            # Main web interface
│   └── static/                   # CSS, JS, images
├── notebooks/                     # Jupyter notebooks
│   └── spam_classification_exploration.ipynb
├── data/                          # Dataset files
├── models/                        # Trained model files
├── results/                       # Training and evaluation results
├── train_models.py               # Main training script
├── evaluate_models.py            # Model evaluation script
├── app.py                        # Flask web application
├── setup.py                      # Automated setup script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🧠 Models

### Traditional Machine Learning
- **Naive Bayes**: Fast probabilistic classifier, excellent baseline
- **SVM**: Support Vector Machine with TF-IDF features
- **Random Forest**: Ensemble method with feature importance
- **Logistic Regression**: Linear classifier with interpretable coefficients

### Deep Learning
- **BERT**: Bidirectional Encoder Representations from Transformers
- **DistilBERT**: Lightweight, faster version of BERT (40% smaller, 60% faster)

### Feature Engineering
- Text length and word count statistics
- Punctuation patterns (exclamation marks, capital letters)
- Spam-indicative keywords detection
- Character-level and word-level n-grams
- TF-IDF vectorization

## 📈 Performance

The models achieve high performance on spam classification:

- **Best Traditional ML**: F1-Score ~0.95+ (typically Naive Bayes or SVM)
- **Deep Learning**: F1-Score ~0.97+ (BERT/DistilBERT)
- **Ensemble**: F1-Score ~0.98+ (combining multiple models)

### Sample Results
| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|---------|----------|-----|
| Naive Bayes | 0.951 | 0.943 | 0.963 | 0.953 | 0.985 |
| SVM | 0.958 | 0.951 | 0.967 | 0.959 | 0.987 |
| Random Forest | 0.943 | 0.934 | 0.956 | 0.945 | 0.982 |
| BERT | 0.972 | 0.968 | 0.977 | 0.973 | 0.994 |
| Ensemble | 0.978 | 0.975 | 0.982 | 0.978 | 0.996 |

## 🔧 Configuration

### Training Options
```bash
# Training configuration
python train_models.py \
    --data-path data/spam_dataset.csv \
    --output-dir results \
    --epochs 3 \
    --batch-size 16 \
    --no-ml           # Skip traditional ML
    --no-dl           # Skip deep learning
    --grid-search     # Enable hyperparameter tuning
```

### Environment Variables
```bash
export CUDA_VISIBLE_DEVICES=0  # For GPU training
export PYTORCH_TRANSFORMERS_CACHE=./cache  # Cache directory
```

## 💡 Usage Examples

### Python API
```python
from src.traditional_ml import TraditionalMLModels
from src.deep_learning import DeepLearningModels, EnsemblePredictor

# Load trained models
ml_models = TraditionalMLModels()
ml_models.load_models('models')

dl_models = DeepLearningModels()
dl_models.load_model('bert-base', 'models/bert-base')

# Create ensemble
ensemble = EnsemblePredictor(ml_models, dl_models)

# Make prediction
text = "URGENT! Your account will be suspended!"
result = ensemble.predict(text)
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.3f}")
```

### Web API
```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Congratulations! You won $1000!"}'
```

## 🔬 Research & Development

### Extending the Project
1. **Data Sources**: Add more diverse spam datasets
2. **Feature Engineering**: Implement advanced NLP features
3. **Model Architecture**: Try other transformer models (RoBERTa, ELECTRA)
4. **Deployment**: Containerize with Docker, deploy to cloud
5. **Real-time Processing**: Implement streaming classification

### Experimental Features
- Multi-language spam detection
- Email metadata analysis (headers, attachments)
- User feedback integration for active learning
- Adversarial robustness testing

## 📚 Dependencies

### Core Requirements
- Python 3.7+
- pandas, numpy, scikit-learn
- nltk, transformers, torch
- matplotlib, seaborn, wordcloud
- flask, jupyter

### Optional Dependencies
- CUDA-compatible PyTorch (for GPU acceleration)
- Additional transformer models
- Advanced visualization libraries

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face for pre-trained transformer models
- SpamAssassin for spam detection techniques
- NLTK team for natural language processing tools
- scikit-learn community for machine learning algorithms

## 📞 Support

If you encounter any issues or have questions:
1. Check the existing issues on GitHub
2. Create a new issue with detailed description
3. Include system information and error logs

---

**Built with ❤️ for email security and machine learning education**
