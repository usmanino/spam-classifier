#!/usr/bin/env python3
"""
Setup script for the spam email classification project.
Installs dependencies and downloads required data.
"""

import subprocess
import sys
import os
import nltk
from pathlib import Path

def run_command(command, description):
    """Run a system command with error handling."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during {description}:")
        print(f"  Command: {command}")
        print(f"  Error: {e.stderr}")
        return False

def install_requirements():
    """Install Python requirements."""
    requirements_file = "requirements.txt"
    if os.path.exists(requirements_file):
        return run_command(f"{sys.executable} -m pip install -r {requirements_file}", 
                         "Installing Python requirements")
    else:
        print("⚠ requirements.txt not found. Installing basic packages...")
        packages = [
            "pandas", "numpy", "scikit-learn", "nltk", "transformers", 
            "torch", "matplotlib", "seaborn", "flask", "jupyter", "wordcloud"
        ]
        for package in packages:
            run_command(f"{sys.executable} -m pip install {package}", 
                       f"Installing {package}")
        return True

def download_nltk_data():
    """Download required NLTK data."""
    print("\nDownloading NLTK data...")
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        print("✓ NLTK data downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Error downloading NLTK data: {str(e)}")
        return False

def create_directories():
    """Create necessary project directories."""
    print("\nCreating project directories...")
    directories = [
        "data", "models", "results", "web/static", "web/templates", 
        "notebooks", "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    return True

def verify_installation():
    """Verify that key dependencies are working."""
    print("\nVerifying installation...")
    
    try:
        # Test core libraries
        import pandas as pd
        import numpy as np
        import sklearn
        import nltk
        import matplotlib.pyplot as plt
        import seaborn as sns
        import flask
        print("✓ Core libraries imported successfully")
        
        # Test transformers (optional, may not be needed for basic functionality)
        try:
            import transformers
            import torch
            print("✓ Deep learning libraries available")
        except ImportError:
            print("⚠ Deep learning libraries not available (transformers/torch)")
            print("  Traditional ML models will still work")
        
        return True
    
    except ImportError as e:
        print(f"✗ Import error: {str(e)}")
        return False

def main():
    """Main setup function."""
    print("🚀 Setting up Spam Email Classification Project")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("✗ Python 3.7 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python version: {sys.version}")
    
    # Setup steps
    steps = [
        ("Creating directories", create_directories),
        ("Installing requirements", install_requirements),
        ("Downloading NLTK data", download_nltk_data),
        ("Verifying installation", verify_installation),
    ]
    
    success_count = 0
    for description, func in steps:
        if func():
            success_count += 1
        else:
            print(f"⚠ {description} failed but continuing...")
    
    print(f"\n{'='*60}")
    if success_count == len(steps):
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run training: python train_models.py")
        print("2. Start web app: python app.py")
        print("3. Open Jupyter notebook: jupyter notebook notebooks/spam_classification_exploration.ipynb")
    else:
        print(f"⚠ Setup completed with {len(steps) - success_count} warnings")
        print("Some features may not be available")
    
    print(f"\nProject structure:")
    print("├── src/                    # Source code")
    print("├── data/                   # Dataset files")
    print("├── models/                 # Trained models")
    print("├── results/                # Training results")
    print("├── web/                    # Web interface")
    print("├── notebooks/              # Jupyter notebooks")
    print("├── train_models.py         # Training script")
    print("├── evaluate_models.py      # Evaluation script")
    print("└── app.py                  # Web application")

if __name__ == "__main__":
    main()
