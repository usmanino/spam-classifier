#!/usr/bin/env python3
"""
Demo script for spam email classification system.
Provides interactive demonstrations of all system capabilities.
"""

import sys
import os
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append('src')

from data_preprocessing import DataLoader, EmailPreprocessor
from traditional_ml import TraditionalMLModels
from deep_learning import DeepLearningModels, EnsemblePredictor

class SpamClassifierDemo:
    """Interactive demo for spam email classification."""
    
    def __init__(self):
        self.loader = DataLoader()
        self.preprocessor = EmailPreprocessor()
        self.ml_models = TraditionalMLModels()
        self.dl_models = DeepLearningModels()
        self.ensemble = None
        
        # Sample emails for testing
        self.sample_emails = {
            "spam_samples": [
                "WINNER! You've won $1000! Click here to claim your prize NOW! Limited time offer!",
                "URGENT! Your account will be suspended unless you verify your information immediately.",
                "FREE MONEY! No strings attached! Make $500/day working from home! Call now!!!",
                "Congratulations! You're pre-approved for a $5000 loan! Apply now for instant cash!",
                "ALERT: Suspicious activity detected on your account. Verify now or lose access!"
            ],
            "ham_samples": [
                "Hey John, thanks for your help with the project yesterday. The presentation went really well!",
                "Don't forget about mom's birthday next week. Should we plan something together?",
                "The weather is really nice today, perfect for a walk in the park.",
                "Can we meet for coffee tomorrow at 3 PM? I have something important to discuss.",
                "Looking forward to seeing you at the conference next month. Safe travels!"
            ]
        }
    
    def print_header(self, title):
        """Print a formatted header."""
        print(f"\n{'='*60}")
        print(f"{title:^60}")
        print(f"{'='*60}")
    
    def print_section(self, title):
        """Print a formatted section header."""
        print(f"\n{'-'*40}")
        print(f"{title}")
        print(f"{'-'*40}")
    
    def demo_data_preprocessing(self):
        """Demonstrate data preprocessing capabilities."""
        self.print_section("📊 Data Preprocessing Demo")
        
        # Load sample data
        print("Loading sample dataset...")
        df = self.loader.load_spam_dataset()
        print(f"✓ Loaded {len(df)} email samples")
        
        # Show original vs processed text
        sample_text = df.iloc[0]['text']
        processed_text = self.preprocessor.preprocess_text(sample_text)
        
        print(f"\nOriginal text (first 100 chars):")
        print(f"'{sample_text[:100]}...'")
        print(f"\nProcessed text:")
        print(f"'{processed_text[:100]}...'")
        
        # Extract features
        features = self.preprocessor.extract_features(sample_text)
        print(f"\nExtracted features:")
        for key, value in features.items():
            print(f"  {key}: {value}")
        
        # Prepare full dataset
        prepared_df = self.loader.prepare_dataset(df)
        print(f"\n✓ Dataset prepared with {prepared_df.shape[1]} features")
        print(f"Label distribution: {dict(prepared_df['label'].value_counts())}")
        
        return prepared_df
    
    def demo_quick_training(self, prepared_df):
        """Demonstrate quick model training."""
        self.print_section("🏋️ Model Loading & Training Demo")
        
        from sklearn.model_selection import train_test_split
        
        # First try to load existing models
        try:
            self.ml_models.load_models('models')
            print(f"✓ Loaded {len(self.ml_models.trained_models)} existing trained models")
            print(f"Available models: {list(self.ml_models.trained_models.keys())}")
            
            # Use existing models for evaluation
            X = prepared_df['cleaned_text']
            y = prepared_df['target']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            print(f"\nTest set: {len(X_test)} samples")
            print("Evaluating existing models...")
            results = self.ml_models.evaluate_all_models(X_test, y_test)
            
            return results
            
        except Exception as e:
            print(f"No existing models found, training new ones...")
            
            # Split data
            X = prepared_df['cleaned_text']
            y = prepared_df['target']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            print(f"Training set: {len(X_train)} samples")
            print(f"Test set: {len(X_test)} samples")
            
            # Train a simple Naive Bayes model for demo
            print("\nTraining Naive Bayes classifier...")
            start_time = time.time()
            
            model = self.ml_models.train_model(X_train, y_train, 'naive_bayes', use_grid_search=False)
            
            training_time = time.time() - start_time
            print(f"✓ Training completed in {training_time:.2f} seconds")
            
            # Evaluate
            print("\nEvaluating model...")
            results = self.ml_models.evaluate_all_models(X_test, y_test)
            
            for model_name, result in results.items():
                print(f"\n{model_name} Results:")
                print(f"  Accuracy: {result['accuracy']:.4f}")
                print(f"  Precision: {result['precision']:.4f}")
                print(f"  Recall: {result['recall']:.4f}")
                print(f"  F1-Score: {result['f1_score']:.4f}")
            
            return results
    
    def demo_predictions(self):
        """Demonstrate real-time predictions."""
        self.print_section("🎯 Prediction Demo")
        
        if not self.ml_models.trained_models:
            print("⚠ No trained models available. Please run training first.")
            return
        
        print("Testing predictions on sample emails...\n")
        
        # Test spam samples
        print("🚨 SPAM SAMPLES:")
        for i, email in enumerate(self.sample_emails["spam_samples"], 1):
            predictions = self.ml_models.predict(email)
            
            print(f"\n{i}. Email: {email[:80]}...")
            for model_name, pred in predictions.items():
                confidence_emoji = "🔴" if pred['spam_probability'] > 0.7 else "🟡" if pred['spam_probability'] > 0.5 else "🟢"
                print(f"   {model_name}: {pred['prediction'].upper()} {confidence_emoji} ({pred['spam_probability']:.3f})")
        
        # Test ham samples
        print(f"\n✅ HAM SAMPLES:")
        for i, email in enumerate(self.sample_emails["ham_samples"], 1):
            predictions = self.ml_models.predict(email)
            
            print(f"\n{i}. Email: {email[:80]}...")
            for model_name, pred in predictions.items():
                confidence_emoji = "🔴" if pred['spam_probability'] > 0.7 else "🟡" if pred['spam_probability'] > 0.5 else "🟢"
                print(f"   {model_name}: {pred['prediction'].upper()} {confidence_emoji} ({pred['spam_probability']:.3f})")
    
    def demo_interactive_prediction(self):
        """Interactive prediction demo."""
        self.print_section("💬 Interactive Prediction Demo")
        
        if not self.ml_models.trained_models:
            print("⚠ No trained models available. Please run training first.")
            return
        
        print("Enter email text to classify (or 'quit' to exit):")
        
        while True:
            try:
                user_input = input("\n📧 Email text: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! 👋")
                    break
                
                if not user_input:
                    print("Please enter some text.")
                    continue
                
                # Make prediction
                predictions = self.ml_models.predict(user_input)
                
                print(f"\n🔍 Analysis Results:")
                print(f"Text length: {len(user_input)} characters")
                
                for model_name, pred in predictions.items():
                    result_emoji = "🚨" if pred['prediction'] == 'spam' else "✅"
                    confidence_str = f"{pred['confidence']*100:.1f}%"
                    spam_prob_str = f"{pred['spam_probability']*100:.1f}%"
                    
                    print(f"   {model_name.replace('_', ' ').title()}: {pred['prediction'].upper()} {result_emoji}")
                    print(f"      Confidence: {confidence_str} | Spam Probability: {spam_prob_str}")
                
            except KeyboardInterrupt:
                print("\n\nDemo interrupted. Goodbye! 👋")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
    
    def demo_performance_analysis(self, results):
        """Demonstrate performance analysis."""
        self.print_section("📈 Performance Analysis Demo")
        
        if not results:
            print("⚠ No results available for analysis.")
            return
        
        # Create performance comparison
        performance_data = []
        for model_name, result in results.items():
            performance_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1-Score': result['f1_score'],
                'AUC': result['auc_score']
            })
        
        df_performance = pd.DataFrame(performance_data)
        
        print("Performance Summary:")
        print(df_performance.to_string(index=False, float_format='%.4f'))
        
        # Find best model
        best_model = df_performance.loc[df_performance['F1-Score'].idxmax()]
        print(f"\n🏆 Best performing model: {best_model['Model']} (F1-Score: {best_model['F1-Score']:.4f})")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"results/demo_results_{timestamp}.json"
        
        demo_results = {
            'timestamp': datetime.now().isoformat(),
            'performance': performance_data,
            'best_model': {
                'name': best_model['Model'],
                'f1_score': float(best_model['F1-Score']),
                'accuracy': float(best_model['Accuracy'])
            },
            'sample_predictions': {
                'spam_samples': len(self.sample_emails['spam_samples']),
                'ham_samples': len(self.sample_emails['ham_samples'])
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(demo_results, f, indent=2)
        
        print(f"✓ Results saved to: {results_file}")
    
    def demo_feature_importance(self):
        """Demonstrate feature importance analysis."""
        self.print_section("🔍 Feature Importance Demo")
        
        if not self.ml_models.trained_models:
            print("⚠ No trained models available.")
            return
        
        # Find a tree-based model for feature importance
        tree_models = ['random_forest']
        available_tree_model = None
        
        for model_name in tree_models:
            if model_name in self.ml_models.trained_models:
                available_tree_model = model_name
                break
        
        if available_tree_model:
            model = self.ml_models.trained_models[available_tree_model]
            
            try:
                # Get feature names and importances
                feature_names = model.named_steps['tfidf'].get_feature_names_out()
                importances = model.named_steps['classifier'].feature_importances_
                
                # Get top 10 features
                top_indices = np.argsort(importances)[-10:]
                
                print(f"Top 10 most important features ({available_tree_model}):")
                for i, idx in enumerate(reversed(top_indices), 1):
                    feature = feature_names[idx]
                    importance = importances[idx]
                    print(f"{i:2d}. {feature:<20}: {importance:.4f}")
                
            except Exception as e:
                print(f"Could not extract feature importance: {str(e)}")
        else:
            print("No tree-based models available for feature importance analysis.")
    
    def run_full_demo(self):
        """Run the complete demonstration."""
        self.print_header("🚀 Spam Email Classification System Demo")
        
        print("Welcome to the comprehensive spam email classification demo!")
        print("This demonstration will showcase all system capabilities.")
        
        try:
            # 1. Data preprocessing demo
            prepared_df = self.demo_data_preprocessing()
            
            # 2. Quick training demo
            results = self.demo_quick_training(prepared_df)
            
            # 3. Prediction demos
            self.demo_predictions()
            
            # 4. Performance analysis
            self.demo_performance_analysis(results)
            
            # 5. Feature importance
            self.demo_feature_importance()
            
            # 6. Interactive demo (optional)
            print(f"\n{'-'*40}")
            user_choice = input("Would you like to try interactive predictions? (y/n): ").strip().lower()
            if user_choice in ['y', 'yes']:
                self.demo_interactive_prediction()
            
            # Summary
            self.print_header("✨ Demo Complete")
            print("Demo completed successfully! 🎉")
            print("\nWhat you've seen:")
            print("• Data preprocessing and feature extraction")
            print("• Machine learning model training")
            print("• Real-time spam classification")
            print("• Performance analysis and metrics")
            print("• Feature importance analysis")
            
            print("\nNext steps:")
            print("1. Run full training: python train_models.py")
            print("2. Start web interface: python app.py")
            print("3. Explore notebooks: jupyter notebook notebooks/")
            print("4. Evaluate models: python evaluate_models.py")
            
        except Exception as e:
            print(f"\n❌ Demo encountered an error: {str(e)}")
            print("Please check your installation and try again.")
    
    def quick_test(self):
        """Quick system test."""
        self.print_header("⚡ Quick System Test")
        
        try:
            # Test data loading
            print("Testing data loading...")
            df = self.loader.load_spam_dataset()
            print(f"✓ Loaded {len(df)} samples")
            
            # Test preprocessing
            print("Testing preprocessing...")
            sample_text = "FREE MONEY! Click here now!"
            processed = self.preprocessor.preprocess_text(sample_text)
            print(f"✓ Processed text: '{processed[:50]}...'")
            
            # Test model creation
            print("Testing model creation...")
            self.ml_models.create_models()
            print(f"✓ Created {len(self.ml_models.models)} model configurations")
            
            print("\n🎉 Quick test passed! System is working correctly.")
            return True
            
        except Exception as e:
            print(f"\n❌ Quick test failed: {str(e)}")
            return False

def main():
    """Main demo function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Spam Classification System Demo')
    parser.add_argument('--quick', action='store_true', help='Run quick test only')
    parser.add_argument('--interactive-only', action='store_true', help='Run interactive predictions only')
    
    args = parser.parse_args()
    
    demo = SpamClassifierDemo()
    
    if args.quick:
        demo.quick_test()
    elif args.interactive_only:
        # Load existing models if available
        try:
            demo.ml_models.load_models('models')
            demo.demo_interactive_prediction()
        except:
            print("No trained models found. Please run training first.")
    else:
        demo.run_full_demo()

if __name__ == "__main__":
    main()