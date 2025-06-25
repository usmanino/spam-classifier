import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

class TraditionalMLModels:
    """Traditional machine learning models for spam classification."""
    
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.trained_models = {}
        
    def create_models(self):
        """Create ML model configurations."""
        self.models = {
            'naive_bayes': {
                'model': MultinomialNB(),
                'params': {
                    'alpha': [0.1, 1.0, 10.0]
                }
            },
            'svm': {
                'model': SVC(probability=True, random_state=42),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            }
        }
        
    def create_pipelines(self):
        """Create ML pipelines with different vectorizers."""
        pipelines = {}
        
        # TF-IDF Word-based pipeline
        for model_name, model_config in self.models.items():
            pipelines[f'{model_name}_tfidf'] = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    stop_words='english',
                    lowercase=True
                )),
                ('classifier', model_config['model'])
            ])
        
        return pipelines
    
    def train_model(self, X_train, y_train, model_name, use_grid_search=True):
        """Train a single model with optional hyperparameter tuning."""
        print(f"Training {model_name}...")
        
        # Create pipeline
        pipelines = self.create_pipelines()
        pipeline = pipelines.get(f'{model_name}_tfidf')
        
        if pipeline is None:
            raise ValueError(f"Model {model_name} not found")
        
        if use_grid_search and model_name in self.models:
            # Grid search for hyperparameter tuning
            param_grid = {}
            for param, values in self.models[model_name]['params'].items():
                param_grid[f'classifier__{param}'] = values
            
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            print(f"Best parameters for {model_name}: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            
        else:
            # Train without grid search
            best_model = pipeline
            best_model.fit(X_train, y_train)
        
        self.trained_models[model_name] = best_model
        return best_model
    
    def train_all_models(self, X_train, y_train, use_grid_search=True):
        """Train all traditional ML models."""
        self.create_models()
        results = {}
        
        for model_name in self.models.keys():
            try:
                model = self.train_model(X_train, y_train, model_name, use_grid_search)
                results[model_name] = model
                print(f"✓ {model_name} training completed")
            except Exception as e:
                print(f"✗ Error training {model_name}: {str(e)}")
                
        return results
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a single model."""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        return {
            'model_name': model_name,
            'classification_report': report,
            'confusion_matrix': cm,
            'auc_score': auc_score,
            'accuracy': report['accuracy'],
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score']
        }
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all trained models."""
        results = {}
        
        for model_name, model in self.trained_models.items():
            try:
                evaluation = self.evaluate_model(model, X_test, y_test, model_name)
                results[model_name] = evaluation
                
                print(f"\n{model_name.upper()} Results:")
                print(f"Accuracy: {evaluation['accuracy']:.4f}")
                print(f"Precision: {evaluation['precision']:.4f}")
                print(f"Recall: {evaluation['recall']:.4f}")
                print(f"F1-Score: {evaluation['f1_score']:.4f}")
                print(f"AUC: {evaluation['auc_score']:.4f}")
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {str(e)}")
        
        return results
    
    def save_models(self, save_dir='models'):
        """Save trained models to disk."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for model_name, model in self.trained_models.items():
            model_path = os.path.join(save_dir, f'{model_name}_model.joblib')
            joblib.dump(model, model_path)
            print(f"Saved {model_name} to {model_path}")
    
    def load_models(self, save_dir='models'):
        """Load trained models from disk."""
        loaded_models = {}
        
        if not os.path.exists(save_dir):
            print(f"Model directory {save_dir} does not exist")
            return loaded_models
        
        for filename in os.listdir(save_dir):
            if filename.endswith('_model.joblib'):
                model_name = filename.replace('_model.joblib', '')
                model_path = os.path.join(save_dir, filename)
                try:
                    model = joblib.load(model_path)
                    loaded_models[model_name] = model
                    print(f"Loaded {model_name} from {model_path}")
                except Exception as e:
                    print(f"Error loading {model_name}: {str(e)}")
        
        self.trained_models.update(loaded_models)
        return loaded_models
    
    def predict(self, text, model_name=None):
        """Make predictions on new text."""
        if model_name:
            if model_name not in self.trained_models:
                raise ValueError(f"Model {model_name} not found")
            models_to_use = {model_name: self.trained_models[model_name]}
        else:
            models_to_use = self.trained_models
        
        predictions = {}
        
        for name, model in models_to_use.items():
            try:
                pred = model.predict([text])[0]
                pred_proba = model.predict_proba([text])[0] if hasattr(model, 'predict_proba') else [1-pred, pred]
                
                predictions[name] = {
                    'prediction': 'spam' if pred == 1 else 'ham',
                    'confidence': max(pred_proba),
                    'spam_probability': pred_proba[1]
                }
            except Exception as e:
                print(f"Error predicting with {name}: {str(e)}")
        
        return predictions

def compare_models(evaluation_results):
    """Compare model performance."""
    comparison_df = pd.DataFrame([
        {
            'Model': result['model_name'],
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1-Score': result['f1_score'],
            'AUC': result['auc_score']
        }
        for result in evaluation_results.values()
    ])
    
    # Sort by F1-score
    comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
    
    return comparison_df

if __name__ == "__main__":
    # Example usage
    from data_preprocessing import DataLoader
    
    # Load and prepare data
    loader = DataLoader()
    df = loader.load_spam_dataset()
    prepared_df = loader.prepare_dataset(df)
    
    # Split data
    X = prepared_df['cleaned_text']
    y = prepared_df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train models
    ml_models = TraditionalMLModels()
    ml_models.train_all_models(X_train, y_train, use_grid_search=False)
    
    # Evaluate models
    results = ml_models.evaluate_all_models(X_test, y_test)
    
    # Compare models
    comparison = compare_models(results)
    print("\nModel Comparison:")
    print(comparison)
