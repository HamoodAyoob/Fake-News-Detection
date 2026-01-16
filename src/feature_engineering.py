"""
Feature Engineering Module for Fake News Detection
Handles TF-IDF vectorization and feature combination
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import joblib
import os


class FeatureEngineer:
    """Class to handle feature engineering operations"""
    
    def __init__(self):
        """Initialize feature engineer with TF-IDF vectorizer and scaler"""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # unigrams and bigrams
            min_df=2,
            max_df=0.8,
            sublinear_tf=True
        )
        self.scaler = StandardScaler()
        self.statistical_features = [
            'text_length', 'word_count', 'avg_word_length',
            'exclamation_count', 'question_count', 'period_count',
            'punctuation_count', 'capital_ratio',
            'sentiment_polarity', 'sentiment_subjectivity'
        ]
        
    def extract_tfidf_features(self, texts: pd.Series, fit: bool = True) -> np.ndarray:
        """
        Extract TF-IDF features from text
        
        Args:
            texts: Series of processed text
            fit: Whether to fit the vectorizer (True for training, False for testing)
            
        Returns:
            TF-IDF feature matrix
        """
        # Remove any NaN or empty strings
        texts = texts.fillna('')
        texts = texts.replace('', 'empty')  # Replace empty strings with placeholder
        
        if fit:
            print("Fitting TF-IDF vectorizer...")
            tfidf_features = self.tfidf_vectorizer.fit_transform(texts)
            print(f"TF-IDF features shape: {tfidf_features.shape}")
        else:
            print("Transforming with existing TF-IDF vectorizer...")
            tfidf_features = self.tfidf_vectorizer.transform(texts)
            
        return tfidf_features.toarray()
    
    def extract_statistical_features(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Extract and normalize statistical features
        
        Args:
            df: DataFrame containing statistical features
            fit: Whether to fit the scaler (True for training, False for testing)
            
        Returns:
            Normalized statistical feature matrix
        """
        # Select statistical features
        stat_features = df[self.statistical_features].values
        
        # Replace any NaN or inf values with 0
        stat_features = np.nan_to_num(stat_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        if fit:
            print("Fitting scaler for statistical features...")
            normalized_features = self.scaler.fit_transform(stat_features)
        else:
            print("Transforming with existing scaler...")
            normalized_features = self.scaler.transform(stat_features)
        
        # Final check for NaN
        normalized_features = np.nan_to_num(normalized_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"Statistical features shape: {normalized_features.shape}")
        
        return normalized_features
    
    def combine_features(self, tfidf_features: np.ndarray, 
                        stat_features: np.ndarray) -> np.ndarray:
        """
        Combine TF-IDF and statistical features
        
        Args:
            tfidf_features: TF-IDF feature matrix
            stat_features: Statistical feature matrix
            
        Returns:
            Combined feature matrix
        """
        combined = np.hstack([tfidf_features, stat_features])
        print(f"Combined features shape: {combined.shape}")
        
        return combined
    
    def engineer_features(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Complete feature engineering pipeline
        
        Args:
            df: Preprocessed DataFrame
            fit: Whether to fit transformers (True for training, False for testing)
            
        Returns:
            Final feature matrix ready for modeling
        """
        print("\nStarting feature engineering...")
        
        # Extract TF-IDF features
        tfidf_features = self.extract_tfidf_features(df['processed_content'], fit=fit)
        
        # Extract statistical features
        stat_features = self.extract_statistical_features(df, fit=fit)
        
        # Combine features
        final_features = self.combine_features(tfidf_features, stat_features)
        
        print("Feature engineering completed!")
        
        return final_features
    
    def save_vectorizer(self, path: str):
        """
        Save TF-IDF vectorizer to file
        
        Args:
            path: Path to save the vectorizer
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.tfidf_vectorizer, path)
        print(f"Vectorizer saved to: {path}")
    
    def load_vectorizer(self, path: str):
        """
        Load TF-IDF vectorizer from file
        
        Args:
            path: Path to load the vectorizer from
        """
        self.tfidf_vectorizer = joblib.load(path)
        print(f"Vectorizer loaded from: {path}")
    
    def save_scaler(self, path: str):
        """
        Save scaler to file
        
        Args:
            path: Path to save the scaler
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.scaler, path)
        print(f"Scaler saved to: {path}")
    
    def load_scaler(self, path: str):
        """
        Load scaler from file
        
        Args:
            path: Path to load the scaler from
        """
        self.scaler = joblib.load(path)
        print(f"Scaler loaded from: {path}")
    
    def get_feature_names(self) -> list:
        """
        Get names of all features
        
        Returns:
            List of feature names
        """
        tfidf_names = self.tfidf_vectorizer.get_feature_names_out().tolist()
        all_names = tfidf_names + self.statistical_features
        return all_names


def main():
    """Main function to demonstrate feature engineering"""
    # Load preprocessed data
    print("Loading preprocessed data...")
    df = pd.read_csv('data/processed/cleaned_data.csv')
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Engineer features
    features = engineer.engineer_features(df, fit=True)
    
    # Save vectorizer
    engineer.save_vectorizer('models/tfidf_vectorizer.pkl')
    engineer.save_scaler('models/scaler.pkl')
    
    print(f"\nFinal feature matrix shape: {features.shape}")
    print("Feature engineering pipeline completed!")


if __name__ == "__main__":
    main()