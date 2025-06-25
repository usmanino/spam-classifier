#!/usr/bin/env python3
"""
Comprehensive training script for spam email classification.
Trains both traditional ML and deep learning models.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataLoader
from traditional_ml import TraditionalMLModels, compare_models
from deep_learning import DeepLearningModels, EnsemblePredictor

class SpamClassifierTrainer:
    """Main trainer class for spam classification models."""
    
    def __init__(self, data_path=None, output_dir='results'):
        self.data_path = data_path
        self.output_dir = output_dir
        self.loader = DataLoader()
        self.ml_models = TraditionalMLModels()
        self.dl_models = DeepLearningModels()
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset."""
        print("Loading and preparing dataset...")
        
        # Load dataset
        if self.data_path:
            df = self.loader.load_spam_dataset(self.data_path)
        else:
            df = self.loader.load_spam_dataset()  # Uses sample data
        
        # Prepare dataset
        prepared_df = self.loader.prepare_dataset(df)
        
        print(f"Dataset shape: {prepared_df.shape}")
        print(f"Label distribution:\n{prepared_df['label'].value_counts()}")
        
        return prepared_df
    
    def split_data(self, df, test_size=0.2, val_size=0.2):
        """Split data into train, validation, and test sets."""
        print("Splitting data...")
        
        # Features and targets
        X_text = df['text']  # Original text for deep learning
        X_cleaned = df['cleaned_text']  # Cleaned text for traditional ML
        y = df['target']
        
        # Additional features
        feature_cols = ['length', 'num_words', 'num_sentences', 'num_exclamation', 
                       'num_question', 'num_uppercase', 'num_digits', 'has_money_words', 
                       'has_urgent_words', 'capital_ratio']
        X_features = df[feature_cols]
        
        # Split data
        X_train_text, X_temp_text, X_train_cleaned, X_temp_cleaned, \
        X_train_features, X_temp_features, y_train, y_temp = train_test_split(
            X_text, X_cleaned, X_features, y, 
            test_size=(test_size + val_size), 
            random_state=42, 
            stratify=y
        )
        
        if val_size > 0:
            val_ratio = val_size / (test_size + val_size)
            X_val_text, X_test_text, X_val_cleaned, X_test_cleaned, \
            X_val_features, X_test_features, y_val, y_test = train_test_split(
                X_temp_text, X_temp_cleaned, X_temp_features, y_temp,
                test_size=(1 - val_ratio),
                random_state=42,
                stratify=y_temp
            )
        else:
            X_test_text, X_test_cleaned, X_test_features = X_temp_text, X_temp_cleaned, X_temp_features
            y_test = y_temp
            X_val_text, X_val_cleaned, X_val_features, y_val = None, None, None, None
        
        print(f"Train set size: {len(X_train_text)}")
        if val_size > 0:
            print(f"Validation set size: {len(X_val_text)}")
        print(f"Test set size: {len(X_test_text)}")
        
        return {
            'train': (X_train_text, X_train_cleaned, X_train_features, y_train),
            'val': (X_val_text, X_val_cleaned, X_val_features, y_val) if val_size > 0 else None,
            'test': (X_test_text, X_test_cleaned, X_test_features, y_test)
        }
    
    def train_traditional_models(self, data_splits, use_grid_search=True):
        """Train traditional ML models."""
        print("\n" + "="*50)
        print("Training Traditional ML Models")
        print("="*50)
        
        X_train_cleaned = data_splits['train'][1]
        y_train = data_splits['train'][3]
        
        # Train models
        self.ml_models.train_all_models(X_train_cleaned, y_train, use_grid_search)
        
        # Save models
        self.ml_models.save_models('models')
        
        return self.ml_models
    
    def train_deep_learning_models(self, data_splits, epochs=3, batch_size=16):
        """Train deep learning models."""
        print("\n" + "="*50)
        print("Training Deep Learning Models")
        print("="*50)
        
        X_train_text = data_splits['train'][0]
        y_train = data_splits['train'][3]
        
        if data_splits['val']:
            X_val_text = data_splits['val'][0]
            y_val = data_splits['val'][3]
        else:
            # Use a portion of training data for validation
            X_train_text, X_val_text, y_train, y_val = train_test_split(
                X_train_text, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
        
        # Initialize models
        self.dl_models.initialize_models()
        
        # Train models
        self.dl_models.train_all_models(
            X_train_text, X_val_text, y_train, y_val,
            epochs=epochs, batch_size=batch_size
        )
        
        return self.dl_models
    
    def evaluate_all_models(self, data_splits):
        """Evaluate all trained models."""
        print("\n" + "="*50)
        print("Evaluating All Models")
        print("="*50)
        
        X_test_text = data_splits['test'][0]
        X_test_cleaned = data_splits['test'][1]
        y_test = data_splits['test'][3]
        
        results = {}
        
        # Evaluate traditional ML models
        if self.ml_models.trained_models:
            print("\nTraditional ML Models:")
            ml_results = self.ml_models.evaluate_all_models(X_test_cleaned, y_test)
            results.update(ml_results)
        
        # Evaluate deep learning models
        if self.dl_models.trained_models:
            print("\nDeep Learning Models:")
            dl_results = self.dl_models.evaluate_all_models(X_test_text, y_test)
            results.update(dl_results)
        
        return results
    
    def create_visualizations(self, results):
        """Create performance visualizations."""
        print("\nCreating visualizations...")
        
        # Prepare data for plotting
        model_names = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        auc_scores = []
        
        for model_name, result in results.items():
            model_names.append(model_name.replace('_', ' ').title())
            accuracies.append(result['accuracy'])
            precisions.append(result['precision'])
            recalls.append(result['recall'])
            f1_scores.append(result['f1_score'])
            auc_scores.append(result['auc_score'])
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics = [
            ('Accuracy', accuracies),
            ('Precision', precisions),
            ('Recall', recalls),
            ('F1-Score', f1_scores),
            ('AUC Score', auc_scores)
        ]
        
        for i, (metric_name, values) in enumerate(metrics):
            row = i // 3
            col = i % 3
            
            bars = axes[row, col].bar(model_names, values, color=plt.cm.viridis(np.linspace(0, 1, len(values))))
            axes[row, col].set_title(metric_name, fontweight='bold')
            axes[row, col].set_ylabel(metric_name)
            axes[row, col].tick_params(axis='x', rotation=45)
            axes[row, col].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[row, col].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                                  f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Remove the last subplot
        fig.delaxes(axes[1, 2])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create detailed comparison table
        comparison_df = pd.DataFrame({
            'Model': model_names,
            'Accuracy': accuracies,
            'Precision': precisions,
            'Recall': recalls,
            'F1-Score': f1_scores,
            'AUC': auc_scores
        })
        
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        
        # Save comparison table
        comparison_df.to_csv(os.path.join(self.output_dir, 'model_comparison.csv'), index=False)
        
        print(f"Best performing model: {comparison_df.iloc[0]['Model']} (F1-Score: {comparison_df.iloc[0]['F1-Score']:.4f})")
        
        return comparison_df
    
    def save_results(self, results, comparison_df):
        """Save training results."""
        print("Saving results...")
        
        # Save detailed results
        results_summary = {
            'timestamp': datetime.now().isoformat(),
            'models_trained': list(results.keys()),
            'best_model': comparison_df.iloc[0]['Model'],
            'best_f1_score': comparison_df.iloc[0]['F1-Score'],
            'results': {
                model: {
                    'accuracy': float(result['accuracy']),
                    'precision': float(result['precision']),
                    'recall': float(result['recall']),
                    'f1_score': float(result['f1_score']),
                    'auc_score': float(result['auc_score'])
                }
                for model, result in results.items()
            }
        }
        
        with open(os.path.join(self.output_dir, 'training_results.json'), 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"Results saved to {self.output_dir}/")
    
    def demo_predictions(self, sample_texts=None):
        """Demonstrate predictions on sample texts."""
        print("\n" + "="*50)
        print("Demo Predictions")
        print("="*50)
        
        if sample_texts is None:
            sample_texts = [
                "Congratulations! You've won $1000! Click here to claim now!",
                "Hey, are we still meeting for coffee tomorrow?",
                "URGENT! Your account will be suspended unless you verify immediately!",
                "Thanks for the great presentation today. The team loved it.",
                "FREE MONEY! No strings attached! Call 1-800-GET-RICH now!"
            ]
        
        # Create ensemble predictor
        ensemble = EnsemblePredictor(self.ml_models, self.dl_models)
        
        for i, text in enumerate(sample_texts, 1):
            print(f"\n--- Sample {i} ---")
            print(f"Text: {text}")
            
            try:
                prediction = ensemble.predict(text)
                print(f"Prediction: {prediction['prediction'].upper()}")
                print(f"Confidence: {prediction['confidence']:.3f}")
                print(f"Spam Probability: {prediction['spam_probability']:.3f}")
            except Exception as e:
                print(f"Error making prediction: {str(e)}")
    
    def run_full_training(self, train_ml=True, train_dl=True, epochs=3, batch_size=16, use_grid_search=False):
        """Run the complete training pipeline."""
        print("Starting Spam Classification Training Pipeline")
        print("=" * 60)
        
        # Load and prepare data
        df = self.load_and_prepare_data()
        
        # Split data
        data_splits = self.split_data(df)
        
        # Train models
        if train_ml:
            self.train_traditional_models(data_splits, use_grid_search)
        
        if train_dl:
            self.train_deep_learning_models(data_splits, epochs, batch_size)
        
        # Evaluate models
        results = self.evaluate_all_models(data_splits)
        
        if results:
            # Create visualizations
            comparison_df = self.create_visualizations(results)
            
            # Save results
            self.save_results(results, comparison_df)
            
            # Demo predictions
            self.demo_predictions()
            
            print("\n" + "="*60)
            print("Training Pipeline Completed Successfully!")
            print("="*60)
            
            return results, comparison_df
        else:
            print("No models were trained successfully.")
            return None, None

def main():
    """Main function to run the training pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train spam email classification models')
    parser.add_argument('--data-path', type=str, help='Path to dataset file')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs for deep learning models')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--no-ml', action='store_true', help='Skip traditional ML training')
    parser.add_argument('--no-dl', action='store_true', help='Skip deep learning training')
    parser.add_argument('--grid-search', action='store_true', help='Use grid search for ML models')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = SpamClassifierTrainer(args.data_path, args.output_dir)
    
    # Run training
    results, comparison = trainer.run_full_training(
        train_ml=not args.no_ml,
        train_dl=not args.no_dl,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_grid_search=args.grid_search
    )
    
    if results:
        print(f"\nTraining completed! Check {args.output_dir}/ for results.")
    else:
        print("Training failed!")

if __name__ == "__main__":
    main()
