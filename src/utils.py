"""
Utility functions for Fake News Detection System
"""

import os
import sys
from pathlib import Path


def check_file_exists(file_path: str) -> bool:
    """
    Check if a file exists
    
    Args:
        file_path: Path to file
        
    Returns:
        True if file exists, False otherwise
    """
    return os.path.exists(file_path)


def create_directory(dir_path: str):
    """
    Create directory if it doesn't exist
    
    Args:
        dir_path: Path to directory
    """
    os.makedirs(dir_path, exist_ok=True)
    print(f"Directory created/verified: {dir_path}")


def setup_project_structure():
    """Create all required directories for the project"""
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'notebooks',
        'src',
        'app/assets',
        'tests'
    ]
    
    print("Setting up project structure...")
    for directory in directories:
        create_directory(directory)
    
    print("✅ Project structure setup complete!")


def check_dataset_exists() -> bool:
    """
    Check if dataset files exist
    
    Returns:
        True if both dataset files exist
    """
    fake_exists = check_file_exists('data/raw/Fake.csv')
    true_exists = check_file_exists('data/raw/True.csv')
    
    return fake_exists and true_exists


def check_model_exists() -> bool:
    """
    Check if trained model exists
    
    Returns:
        True if model file exists
    """
    return check_file_exists('models/fake_news_model.pkl')


def check_vectorizer_exists() -> bool:
    """
    Check if vectorizer exists
    
    Returns:
        True if vectorizer file exists
    """
    return check_file_exists('models/tfidf_vectorizer.pkl')


def check_processed_data_exists() -> bool:
    """
    Check if processed data exists
    
    Returns:
        True if processed data file exists
    """
    return check_file_exists('data/processed/cleaned_data.csv')


def print_banner(text: str):
    """
    Print formatted banner
    
    Args:
        text: Text to display in banner
    """
    print("\n" + "="*60)
    print(text.center(60))
    print("="*60 + "\n")


def print_section(text: str):
    """
    Print formatted section header
    
    Args:
        text: Section header text
    """
    print("\n" + "-"*60)
    print(text)
    print("-"*60)


def get_project_root() -> Path:
    """
    Get project root directory
    
    Returns:
        Path object for project root
    """
    return Path(__file__).parent.parent


def print_file_tree():
    """Print expected file structure"""
    tree = """
    fake-news-detection/
    │
    ├── data/
    │   ├── raw/
    │   │   ├── Fake.csv
    │   │   └── True.csv
    │   │
    │   └── processed/
    │       ├── cleaned_data.csv
    │       └── data_stats.txt
    │
    ├── models/
    │   ├── fake_news_model.pkl
    │   ├── tfidf_vectorizer.pkl
    │   └── model_metrics.txt
    │
    ├── src/
    │   ├── data_preprocessing.py
    │   ├── feature_engineering.py
    │   ├── model_training.py
    │   ├── model_evaluation.py
    │   └── utils.py
    │
    ├── app/
    │   └── streamlit_app.py
    │
    └── run_project.py
    """
    print(tree)


def validate_environment():
    """Validate Python environment and dependencies"""
    print("Validating environment...")
    
    # Check Python version
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher required!")
        return False
    
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    
    # Check required packages
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'nltk', 'textblob',
        'matplotlib', 'seaborn', 'wordcloud', 'joblib',
        'streamlit', 'plotly', 'xgboost'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            print(f"❌ {package} not found")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✅ All required packages installed!")
    return True


def display_download_instructions():
    """Display instructions for downloading dataset"""
    print_banner("DATASET DOWNLOAD INSTRUCTIONS")
    
    print("""
    📥 How to Download the Dataset:
    
    1. Visit Kaggle dataset page:
       https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
    
    2. Sign in to your Kaggle account (create one if needed)
    
    3. Click the "Download" button to get the dataset
    
    4. Extract the downloaded ZIP file
    
    5. You should see two files:
       - Fake.csv
       - True.csv
    
    6. Place these files in the following directory:
       fake-news-detection/data/raw/
    
    7. Your folder structure should look like:
       data/
       └── raw/
           ├── Fake.csv
           └── True.csv
    
    ✅ Once files are in place, return to the menu and continue!
    """)


if __name__ == "__main__":
    # Test utilities
    print_banner("TESTING UTILITIES")
    setup_project_structure()
    print("\n")
    print_file_tree()