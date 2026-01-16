"""
Data Preprocessing Module for Fake News Detection
Handles loading, cleaning, and preprocessing of news articles
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import os
from tqdm import tqdm

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class DataPreprocessor:
    """Class to handle all data preprocessing operations"""
    
    def __init__(self):
        """Initialize preprocessor with required NLP tools"""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def load_data(self, fake_path: str, true_path: str) -> pd.DataFrame:
        """
        Load fake and true news datasets and combine them
        
        Args:
            fake_path: Path to Fake.csv
            true_path: Path to True.csv
            
        Returns:
            Combined DataFrame with labels
        """
        print("Loading datasets...")
        
        # Load fake news
        fake_df = pd.read_csv(fake_path)
        fake_df['label'] = 0  # 0 for fake
        
        # Load true news
        true_df = pd.read_csv(true_path)
        true_df['label'] = 1  # 1 for real/true
        
        # Combine datasets
        df = pd.concat([fake_df, true_df], ignore_index=True)
        
        print(f"Loaded {len(fake_df)} fake news articles")
        print(f"Loaded {len(true_df)} real news articles")
        print(f"Total articles: {len(df)}")
        
        return df
    
    def clean_text(self, text: str) -> str:
        """
        Clean individual text by removing unwanted characters and patterns
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions (@username)
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags
        text = re.sub(r'#\w+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text: str) -> str:
        """
        Tokenize text, remove stopwords, and apply lemmatization
        
        Args:
            text: Cleaned text string
            
        Returns:
            Processed text string
        """
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(processed_tokens)
    
    def extract_features(self, text: str, original_text: str) -> dict:
        """
        Extract statistical features from text
        
        Args:
            text: Processed text
            original_text: Original text before cleaning
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Text length (character count)
        features['text_length'] = len(original_text)
        
        # Word count
        features['word_count'] = len(text.split())
        
        # Average word length
        words = text.split()
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        
        # Punctuation counts
        features['exclamation_count'] = original_text.count('!')
        features['question_count'] = original_text.count('?')
        features['period_count'] = original_text.count('.')
        features['punctuation_count'] = sum([
            original_text.count(p) for p in '!?.,'
        ])
        
        # Capital letter ratio
        capitals = sum(1 for c in original_text if c.isupper())
        features['capital_ratio'] = capitals / len(original_text) if len(original_text) > 0 else 0
        
        # Sentiment analysis using TextBlob
        blob = TextBlob(original_text)
        features['sentiment_polarity'] = blob.sentiment.polarity
        features['sentiment_subjectivity'] = blob.sentiment.subjectivity
        
        return features
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete preprocessing pipeline
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        print("\nStarting preprocessing pipeline...")
        
        # Handle missing values
        df = df.dropna(subset=['title', 'text'])
        print(f"After removing missing values: {len(df)} articles")
        
        # Combine title and text into content
        df['content'] = df['title'] + ' ' + df['text']
        
        # Remove duplicates based on content
        df = df.drop_duplicates(subset=['content'], keep='first')
        print(f"After removing duplicates: {len(df)} articles")
        
        # Store original content for feature extraction
        df['original_content'] = df['content']
        
        # Clean text
        print("Cleaning text...")
        tqdm.pandas(desc="Cleaning")
        df['cleaned_content'] = df['content'].progress_apply(self.clean_text)
        
        # Tokenize and lemmatize
        print("Tokenizing and lemmatizing...")
        tqdm.pandas(desc="Processing")
        df['processed_content'] = df['cleaned_content'].progress_apply(self.tokenize_and_lemmatize)
        
        # Remove rows where processed_content is empty or too short
        df['processed_content'] = df['processed_content'].replace('', np.nan)
        df = df.dropna(subset=['processed_content'])
        df = df[df['processed_content'].str.len() > 10]  # At least 10 characters
        print(f"After removing empty texts: {len(df)} articles")
        
        # Extract features
        print("Extracting features...")
        tqdm.pandas(desc="Feature extraction")
        feature_dicts = df.progress_apply(
            lambda row: self.extract_features(row['processed_content'], row['original_content']),
            axis=1
        )
        
        # Convert feature dictionaries to DataFrame
        feature_df = pd.DataFrame(list(feature_dicts))
        
        # Combine with original DataFrame
        df = pd.concat([df, feature_df], axis=1)
        
        # Final check for any NaN in processed_content
        df = df.dropna(subset=['processed_content'])
        
        # Shuffle dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print("\nPreprocessing completed!")
        print(f"Final dataset size: {len(df)} articles")
        
        return df
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str):
        """
        Save processed data to CSV
        
        Args:
            df: Processed DataFrame
            output_path: Path to save the processed data
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"\nProcessed data saved to: {output_path}")
        
    def save_statistics(self, df: pd.DataFrame, stats_path: str):
        """
        Save dataset statistics to text file
        
        Args:
            df: Processed DataFrame
            stats_path: Path to save statistics
        """
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        
        with open(stats_path, 'w') as f:
            f.write("="*50 + "\n")
            f.write("DATASET STATISTICS\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Total articles: {len(df)}\n")
            f.write(f"Fake news: {len(df[df['label'] == 0])}\n")
            f.write(f"Real news: {len(df[df['label'] == 1])}\n\n")
            
            f.write("Feature Statistics:\n")
            f.write("-"*50 + "\n")
            
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            f.write(df[numeric_columns].describe().to_string())
            
        print(f"Statistics saved to: {stats_path}")


def main():
    """Main function to run preprocessing"""
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Define paths
    fake_path = 'data/raw/Fake.csv'
    true_path = 'data/raw/True.csv'
    output_path = 'data/processed/cleaned_data.csv'
    stats_path = 'data/processed/data_stats.txt'
    
    # Load data
    df = preprocessor.load_data(fake_path, true_path)
    
    # Preprocess data
    df = preprocessor.preprocess_data(df)
    
    # Save processed data
    preprocessor.save_processed_data(df, output_path)
    
    # Save statistics
    preprocessor.save_statistics(df, stats_path)
    
    print("\n" + "="*50)
    print("Preprocessing completed successfully!")
    print("="*50)


if __name__ == "__main__":
    main()