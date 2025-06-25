import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    BertTokenizer, BertForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification,
    TrainingArguments, Trainer
)
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import json

class SpamDataset(Dataset):
    """Custom dataset for spam classification with transformers."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx]) if hasattr(self.texts, 'iloc') else str(self.texts[idx])
        label = self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class DeepLearningModels:
    """Pre-trained deep learning models for spam classification."""
    
    def __init__(self, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.tokenizers = {}
        self.trained_models = {}
        
        print(f"Using device: {self.device}")
    
    def initialize_models(self):
        """Initialize pre-trained models and tokenizers."""
        model_configs = {
            'bert-base': {
                'model_name': 'bert-base-uncased',
                'tokenizer_class': BertTokenizer,
                'model_class': BertForSequenceClassification
            },
            'distilbert': {
                'model_name': 'distilbert-base-uncased',
                'tokenizer_class': DistilBertTokenizer,
                'model_class': DistilBertForSequenceClassification
            }
        }
        
        for model_key, config in model_configs.items():
            try:
                print(f"Loading {model_key}...")
                tokenizer = config['tokenizer_class'].from_pretrained(config['model_name'])
                model = config['model_class'].from_pretrained(
                    config['model_name'],
                    num_labels=2,
                    output_attentions=False,
                    output_hidden_states=False
                )
                
                self.tokenizers[model_key] = tokenizer
                self.models[model_key] = model
                
                print(f"✓ {model_key} loaded successfully")
                
            except Exception as e:
                print(f"✗ Error loading {model_key}: {str(e)}")
    
    def prepare_datasets(self, X_train, X_val, y_train, y_val, model_key, max_length=512):
        """Prepare datasets for training."""
        tokenizer = self.tokenizers[model_key]
        
        train_dataset = SpamDataset(X_train, y_train, tokenizer, max_length)
        val_dataset = SpamDataset(X_val, y_val, tokenizer, max_length)
        
        return train_dataset, val_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train_model(self, model_key, X_train, X_val, y_train, y_val, 
                   epochs=3, batch_size=16, learning_rate=2e-5, save_dir='models'):
        """Train a single model."""
        if model_key not in self.models:
            raise ValueError(f"Model {model_key} not initialized")
        
        print(f"Training {model_key}...")
        
        model = self.models[model_key].to(self.device)
        
        # Prepare datasets
        train_dataset, val_dataset = self.prepare_datasets(
            X_train, X_val, y_train, y_val, model_key
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f'{save_dir}/{model_key}',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{save_dir}/{model_key}/logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            learning_rate=learning_rate,
            save_total_limit=2,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizers[model_key].save_pretrained(f'{save_dir}/{model_key}')
        
        self.trained_models[model_key] = trainer.model
        
        print(f"✓ {model_key} training completed")
        return trainer
    
    def train_all_models(self, X_train, X_val, y_train, y_val, **kwargs):
        """Train all initialized models."""
        results = {}
        
        for model_key in self.models.keys():
            try:
                trainer = self.train_model(model_key, X_train, X_val, y_train, y_val, **kwargs)
                results[model_key] = trainer
            except Exception as e:
                print(f"Error training {model_key}: {str(e)}")
        
        return results
    
    def evaluate_model(self, model_key, X_test, y_test):
        """Evaluate a single model."""
        if model_key not in self.trained_models:
            raise ValueError(f"Model {model_key} not trained")
        
        model = self.trained_models[model_key]
        tokenizer = self.tokenizers[model_key]
        
        # Prepare test dataset
        test_dataset = SpamDataset(X_test, y_test, tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        model.eval()
        predictions = []
        true_labels = []
        probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Evaluating {model_key}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                probabilities.extend(probs[:, 1].cpu().numpy())  # Probability of spam class
        
        # Compute metrics
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
        accuracy = accuracy_score(true_labels, predictions)
        auc_score = roc_auc_score(true_labels, probabilities)
        
        return {
            'model_name': model_key,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc_score,
            'predictions': predictions,
            'probabilities': probabilities
        }
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all trained models."""
        results = {}
        
        for model_key in self.trained_models.keys():
            try:
                evaluation = self.evaluate_model(model_key, X_test, y_test)
                results[model_key] = evaluation
                
                print(f"\n{model_key.upper()} Results:")
                print(f"Accuracy: {evaluation['accuracy']:.4f}")
                print(f"Precision: {evaluation['precision']:.4f}")
                print(f"Recall: {evaluation['recall']:.4f}")
                print(f"F1-Score: {evaluation['f1_score']:.4f}")
                print(f"AUC: {evaluation['auc_score']:.4f}")
                
            except Exception as e:
                print(f"Error evaluating {model_key}: {str(e)}")
        
        return results
    
    def load_model(self, model_key, model_path):
        """Load a trained model from disk."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            
            self.tokenizers[model_key] = tokenizer
            self.trained_models[model_key] = model.to(self.device)
            
            print(f"✓ Loaded {model_key} from {model_path}")
            
        except Exception as e:
            print(f"✗ Error loading {model_key}: {str(e)}")
    
    def predict(self, text, model_key=None, max_length=512):
        """Make predictions on new text."""
        if model_key:
            if model_key not in self.trained_models:
                raise ValueError(f"Model {model_key} not found")
            models_to_use = {model_key: (self.trained_models[model_key], self.tokenizers[model_key])}
        else:
            models_to_use = {
                key: (model, self.tokenizers[key]) 
                for key, model in self.trained_models.items()
            }
        
        predictions = {}
        
        for name, (model, tokenizer) in models_to_use.items():
            try:
                model.eval()
                
                # Tokenize input
                encoding = tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=max_length,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)
                    pred = torch.argmax(logits, dim=-1)
                
                predictions[name] = {
                    'prediction': 'spam' if pred.item() == 1 else 'ham',
                    'confidence': max(probs[0]).item(),
                    'spam_probability': probs[0][1].item()
                }
                
            except Exception as e:
                print(f"Error predicting with {name}: {str(e)}")
        
        return predictions

class EnsemblePredictor:
    """Ensemble predictor combining multiple models."""
    
    def __init__(self, ml_models, dl_models):
        self.ml_models = ml_models
        self.dl_models = dl_models
    
    def predict(self, text, weights=None):
        """Make ensemble predictions."""
        # Get predictions from traditional ML models
        ml_predictions = self.ml_models.predict(text)
        
        # Get predictions from deep learning models
        dl_predictions = self.dl_models.predict(text)
        
        # Combine predictions
        all_predictions = {**ml_predictions, **dl_predictions}
        
        if not all_predictions:
            return {'prediction': 'ham', 'confidence': 0.5}
        
        # Calculate weighted average
        if weights is None:
            weights = {model: 1.0 for model in all_predictions.keys()}
        
        total_weight = sum(weights.get(model, 1.0) for model in all_predictions.keys())
        weighted_spam_prob = sum(
            pred['spam_probability'] * weights.get(model, 1.0) 
            for model, pred in all_predictions.items()
        ) / total_weight
        
        final_prediction = 'spam' if weighted_spam_prob > 0.5 else 'ham'
        confidence = max(weighted_spam_prob, 1 - weighted_spam_prob)
        
        return {
            'prediction': final_prediction,
            'confidence': confidence,
            'spam_probability': weighted_spam_prob,
            'individual_predictions': all_predictions
        }

if __name__ == "__main__":
    # Example usage
    from data_preprocessing import DataLoader
    
    # Load and prepare data
    loader = DataLoader()
    df = loader.load_spam_dataset()
    prepared_df = loader.prepare_dataset(df)
    
    # Split data
    X = prepared_df['text']  # Use original text for transformers
    y = prepared_df['target']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # Initialize and train models
    dl_models = DeepLearningModels()
    dl_models.initialize_models()
    
    # Train with smaller epochs for demonstration
    dl_models.train_all_models(X_train, X_val, y_train, y_val, epochs=1, batch_size=8)
    
    # Evaluate models
    results = dl_models.evaluate_all_models(X_test, y_test)
    
    print("\nDeep Learning Model Results:")
    for model_name, result in results.items():
        print(f"{model_name}: F1={result['f1_score']:.4f}, AUC={result['auc_score']:.4f}")
