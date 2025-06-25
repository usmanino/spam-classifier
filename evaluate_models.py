#!/usr/bin/env python3
"""
Model evaluation script for spam email classification.
Evaluates trained models and generates detailed performance reports.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataLoader
from traditional_ml import TraditionalMLModels
from deep_learning import DeepLearningModels

class ModelEvaluator:
    """Comprehensive model evaluation for spam classification."""
    
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        self.loader = DataLoader()
        self.ml_models = TraditionalMLModels()
        self.dl_models = DeepLearningModels()
        
        os.makedirs(output_dir, exist_ok=True)
    
    def load_models(self):
        """Load all trained models."""
        print("Loading trained models...")
        
        # Load traditional ML models
        self.ml_models.load_models('models')
        print(f"Loaded {len(self.ml_models.trained_models)} ML models")
        
        # Load deep learning models
        model_dir = 'models'
        for model_name in ['bert-base', 'distilbert']:
            model_path = os.path.join(model_dir, model_name)
            if os.path.exists(model_path):
                self.dl_models.load_model(model_name, model_path)
        
        print(f"Loaded {len(self.dl_models.trained_models)} DL models")
        
        total_models = len(self.ml_models.trained_models) + len(self.dl_models.trained_models)
        if total_models == 0:
            raise ValueError("No trained models found. Please run training first.")
        
        return total_models
    
    def prepare_test_data(self, data_path=None):
        """Prepare test dataset."""
        print("Preparing test data...")
        
        # Load and prepare dataset
        df = self.loader.load_spam_dataset(data_path)
        prepared_df = self.loader.prepare_dataset(df)
        
        # Split data (use same random state as training)
        X_text = prepared_df['text']
        X_cleaned = prepared_df['cleaned_text']
        y = prepared_df['target']
        
        _, X_test_text, _, X_test_cleaned, _, y_test = train_test_split(
            X_text, X_cleaned, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_test_text, X_test_cleaned, y_test, prepared_df
    
    def evaluate_all_models(self, X_test_text, X_test_cleaned, y_test):
        """Evaluate all models and return detailed results."""
        print("Evaluating all models...")
        
        results = {}
        
        # Evaluate traditional ML models
        if self.ml_models.trained_models:
            print("\nEvaluating Traditional ML Models:")
            ml_results = self.ml_models.evaluate_all_models(X_test_cleaned, y_test)
            results.update(ml_results)
        
        # Evaluate deep learning models
        if self.dl_models.trained_models:
            print("\nEvaluating Deep Learning Models:")
            dl_results = self.dl_models.evaluate_all_models(X_test_text, y_test)
            results.update(dl_results)
        
        return results
    
    def create_confusion_matrices(self, results, y_test):
        """Create confusion matrix visualizations."""
        print("Creating confusion matrices...")
        
        n_models = len(results)
        if n_models == 0:
            return
        
        # Calculate grid dimensions
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (model_name, result) in enumerate(results.items()):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            cm = result['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
            ax.set_title(f'{model_name.replace("_", " ").title()}\nAccuracy: {result["accuracy"]:.3f}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Hide empty subplots
        for idx in range(n_models, rows * cols):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_roc_curves(self, results, X_test_text, X_test_cleaned, y_test):
        """Create ROC curve visualizations."""
        print("Creating ROC curves...")
        
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
        
        for (model_name, result), color in zip(results.items(), colors):
            try:
                # Get model predictions
                if model_name in self.ml_models.trained_models:
                    model = self.ml_models.trained_models[model_name]
                    y_proba = model.predict_proba(X_test_cleaned)[:, 1]
                elif model_name in self.dl_models.trained_models:
                    # Use stored probabilities from evaluation
                    y_proba = result.get('probabilities', [])
                else:
                    continue
                
                if len(y_proba) == 0:
                    continue
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, color=color, lw=2, 
                        label=f'{model_name.replace("_", " ").title()} (AUC = {roc_auc:.3f})')
                
            except Exception as e:
                print(f"Error creating ROC curve for {model_name}: {str(e)}")
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Spam Classification Models')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_performance_comparison(self, results):
        """Create detailed performance comparison."""
        print("Creating performance comparison...")
        
        # Prepare data
        data = []
        for model_name, result in results.items():
            data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Type': 'Traditional ML' if model_name in self.ml_models.trained_models else 'Deep Learning',
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1-Score': result['f1_score'],
                'AUC': result['auc_score']
            })
        
        df = pd.DataFrame(data)
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        
        for i, metric in enumerate(metrics):
            row = i // 3
            col = i % 3
            
            # Sort by metric for better visualization
            sorted_df = df.sort_values(metric, ascending=True)
            
            bars = axes[row, col].barh(sorted_df['Model'], sorted_df[metric], 
                                     color=['#FF6B6B' if t == 'Traditional ML' else '#4ECDC4' 
                                           for t in sorted_df['Type']])
            
            axes[row, col].set_title(f'{metric}', fontweight='bold', fontsize=12)
            axes[row, col].set_xlabel(metric)
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].set_xlim(0, 1)
            
            # Add value labels
            for bar, value in zip(bars, sorted_df[metric]):
                axes[row, col].text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                                  f'{value:.3f}', va='center', fontweight='bold')
        
        # Remove the last subplot
        fig.delaxes(axes[1, 2])
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#FF6B6B', label='Traditional ML'),
                          Patch(facecolor='#4ECDC4', label='Deep Learning')]
        fig.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.85, 0.15))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return df
    
    def analyze_misclassifications(self, results, X_test_text, X_test_cleaned, y_test):
        """Analyze misclassified examples."""
        print("Analyzing misclassifications...")
        
        if not results:
            return
        
        # Use the best performing model for analysis
        best_model_name = max(results.keys(), key=lambda k: results[k]['f1_score'])
        best_result = results[best_model_name]
        
        print(f"Analyzing misclassifications for best model: {best_model_name}")
        
        # Get predictions
        if best_model_name in self.ml_models.trained_models:
            model = self.ml_models.trained_models[best_model_name]
            y_pred = model.predict(X_test_cleaned)
            X_text = X_test_cleaned
        elif best_model_name in self.dl_models.trained_models:
            # For DL models, we need to re-predict or use stored results
            # This is a simplified version - in practice, you'd store predictions
            print("Misclassification analysis for DL models requires stored predictions")
            return
        else:
            return
        
        # Find misclassified examples
        misclassified_mask = y_pred != y_test
        misclassified_indices = np.where(misclassified_mask)[0]
        
        if len(misclassified_indices) == 0:
            print("No misclassifications found!")
            return
        
        # Analyze false positives and false negatives
        false_positives = np.where((y_pred == 1) & (y_test == 0))[0]
        false_negatives = np.where((y_pred == 0) & (y_test == 1))[0]
        
        print(f"Total misclassifications: {len(misclassified_indices)}")
        print(f"False positives (ham classified as spam): {len(false_positives)}")
        print(f"False negatives (spam classified as ham): {len(false_negatives)}")
        
        # Save examples
        misclassified_examples = []
        
        # Sample some false positives
        for idx in false_positives[:5]:
            misclassified_examples.append({
                'type': 'False Positive',
                'actual': 'ham',
                'predicted': 'spam',
                'text': X_test_text.iloc[idx] if hasattr(X_test_text, 'iloc') else X_test_text[idx]
            })
        
        # Sample some false negatives
        for idx in false_negatives[:5]:
            misclassified_examples.append({
                'type': 'False Negative',
                'actual': 'spam',
                'predicted': 'ham',
                'text': X_test_text.iloc[idx] if hasattr(X_test_text, 'iloc') else X_test_text[idx]
            })
        
        # Save to file
        with open(os.path.join(self.output_dir, 'misclassified_examples.json'), 'w') as f:
            json.dump(misclassified_examples, f, indent=2)
        
        return misclassified_examples
    
    def generate_report(self, results, comparison_df, total_samples):
        """Generate comprehensive evaluation report."""
        print("Generating evaluation report...")
        
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'total_models_evaluated': len(results),
            'total_test_samples': int(total_samples),
            'best_model': {
                'name': comparison_df.iloc[0]['Model'],
                'type': comparison_df.iloc[0]['Type'],
                'f1_score': float(comparison_df.iloc[0]['F1-Score']),
                'accuracy': float(comparison_df.iloc[0]['Accuracy']),
                'auc': float(comparison_df.iloc[0]['AUC'])
            },
            'model_performance': {
                model: {
                    'accuracy': float(result['accuracy']),
                    'precision': float(result['precision']),
                    'recall': float(result['recall']),
                    'f1_score': float(result['f1_score']),
                    'auc_score': float(result['auc_score'])
                }
                for model, result in results.items()
            },
            'summary_statistics': {
                'mean_accuracy': float(comparison_df['Accuracy'].mean()),
                'std_accuracy': float(comparison_df['Accuracy'].std()),
                'mean_f1': float(comparison_df['F1-Score'].mean()),
                'std_f1': float(comparison_df['F1-Score'].std()),
                'best_accuracy': float(comparison_df['Accuracy'].max()),
                'best_f1': float(comparison_df['F1-Score'].max())
            }
        }
        
        # Save report
        with open(os.path.join(self.output_dir, 'evaluation_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save comparison table
        comparison_df.to_csv(os.path.join(self.output_dir, 'model_comparison_detailed.csv'), index=False)
        
        print(f"Evaluation report saved to {self.output_dir}/")
        return report
    
    def run_full_evaluation(self, data_path=None):
        """Run complete model evaluation pipeline."""
        print("Starting Model Evaluation Pipeline")
        print("=" * 50)
        
        try:
            # Load models
            total_models = self.load_models()
            
            # Prepare test data
            X_test_text, X_test_cleaned, y_test, full_df = self.prepare_test_data(data_path)
            
            # Evaluate all models
            results = self.evaluate_all_models(X_test_text, X_test_cleaned, y_test)
            
            if not results:
                print("No models could be evaluated!")
                return None
            
            # Create visualizations
            self.create_confusion_matrices(results, y_test)
            self.create_roc_curves(results, X_test_text, X_test_cleaned, y_test)
            comparison_df = self.create_performance_comparison(results)
            
            # Analyze misclassifications
            self.analyze_misclassifications(results, X_test_text, X_test_cleaned, y_test)
            
            # Generate comprehensive report
            report = self.generate_report(results, comparison_df, len(y_test))
            
            print("\n" + "="*50)
            print("Evaluation Pipeline Completed Successfully!")
            print("="*50)
            print(f"Best Model: {report['best_model']['name']} (F1: {report['best_model']['f1_score']:.4f})")
            print(f"Results saved to: {self.output_dir}/")
            
            return results, comparison_df, report
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            return None

def main():
    """Main function to run model evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate spam email classification models')
    parser.add_argument('--data-path', type=str, help='Path to test dataset file')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(args.output_dir)
    
    # Run evaluation
    results = evaluator.run_full_evaluation(args.data_path)
    
    if results:
        print("Evaluation completed successfully!")
    else:
        print("Evaluation failed!")

if __name__ == "__main__":
    main()
