"""
Model Training Module for Fake News Detection
Trains multiple ML models and performs hyperparameter tuning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib
import os
from datetime import datetime
from feature_engineering import FeatureEngineer


class ModelTrainer:
    """Class to handle model training operations"""
    
    def __init__(self):
        """Initialize model trainer with various classifiers"""
        self.models = {
            'Logistic Regression': LogisticRegression(
                solver='liblinear',
                max_iter=1000,
                C=1.0,
                random_state=42
            ),
            'Naive Bayes': MultinomialNB(alpha=1.0),
            'SVM': SVC(
                kernel='linear',
                C=1.0,
                probability=True,
                random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=30,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=7,
                random_state=42,
                n_jobs=-1
            )
        }
        self.best_model = None
        self.best_model_name = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def prepare_data(self, df: pd.DataFrame, engineer: FeatureEngineer, 
                    test_size: float = 0.2):
        """
        Prepare train-test split
        
        Args:
            df: Preprocessed DataFrame
            engineer: FeatureEngineer instance
            test_size: Proportion of test set
        """
        print("\nPreparing data for training...")
        
        # Engineer features
        X = engineer.engineer_features(df, fit=True)
        y = df['label'].values
        
        # Ensure labels are integers
        y = y.astype(int)
        
        # Train-test split with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )
        
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
        print(f"Feature dimensions: {self.X_train.shape[1]}")
        
        # Check class distribution
        train_dist = np.bincount(self.y_train)
        test_dist = np.bincount(self.y_test)
        
        print(f"\nTraining set - Fake: {train_dist[0]}, Real: {train_dist[1]}")
        print(f"Test set - Fake: {test_dist[0]}, Real: {test_dist[1]}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self, model_name: str, model, cv_folds: int = 5):
        """
        Train a single model with cross-validation
        
        Args:
            model_name: Name of the model
            model: Model instance
            cv_folds: Number of CV folds
            
        Returns:
            Trained model and CV scores
        """
        print(f"\nTraining {model_name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, self.X_train, self.y_train,
            cv=cv_folds,
            scoring='accuracy',
            n_jobs=-1
        )
        
        print(f"CV Scores: {cv_scores}")
        print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Train on full training set
        model.fit(self.X_train, self.y_train)
        
        # Evaluate on test set
        train_score = model.score(self.X_train, self.y_train)
        test_score = model.score(self.X_test, self.y_test)
        
        print(f"Training Accuracy: {train_score:.4f}")
        print(f"Test Accuracy: {test_score:.4f}")
        
        return model, cv_scores, test_score
    
    def train_all_models(self):
        """
        Train all models and compare performance
        
        Returns:
            Dictionary of results
        """
        print("\n" + "="*60)
        print("TRAINING ALL MODELS")
        print("="*60)
        
        results = {}
        
        for name, model in self.models.items():
            trained_model, cv_scores, test_score = self.train_model(name, model)
            
            results[name] = {
                'model': trained_model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': test_score
            }
        
        # Find best model based on test accuracy
        self.best_model_name = max(results.keys(), 
                                   key=lambda k: results[k]['test_accuracy'])
        self.best_model = results[self.best_model_name]['model']
        
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        comparison_df = pd.DataFrame({
            'Model': list(results.keys()),
            'CV Mean': [results[k]['cv_mean'] for k in results.keys()],
            'CV Std': [results[k]['cv_std'] for k in results.keys()],
            'Test Accuracy': [results[k]['test_accuracy'] for k in results.keys()]
        })
        
        comparison_df = comparison_df.sort_values('Test Accuracy', ascending=False)
        print(comparison_df.to_string(index=False))
        
        print(f"\nBest Model: {self.best_model_name}")
        print(f"Best Test Accuracy: {results[self.best_model_name]['test_accuracy']:.4f}")
        
        return results
    
    def tune_hyperparameters(self, model_name: str = 'Random Forest'):
        """
        Perform hyperparameter tuning on selected model
        
        Args:
            model_name: Name of model to tune
            
        Returns:
            Best model after tuning
        """
        print(f"\n" + "="*60)
        print(f"HYPERPARAMETER TUNING FOR {model_name}")
        print("="*60)
        
        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [20, 30, 40],
                'min_samples_split': [2, 5, 10]
            }
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
            
        elif model_name == 'Logistic Regression':
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000, 2000]
            }
            base_model = LogisticRegression(random_state=42)
            
        elif model_name == 'XGBoost':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 5, 7]
            }
            base_model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
            
        else:
            print(f"Tuning not configured for {model_name}")
            return None
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        print("Starting grid search...")
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"\nBest Parameters: {grid_search.best_params_}")
        print(f"Best CV Score: {grid_search.best_score_:.4f}")
        
        # Evaluate on test set
        test_score = grid_search.score(self.X_test, self.y_test)
        print(f"Test Accuracy: {test_score:.4f}")
        
        return grid_search.best_estimator_
    
    def save_model(self, model, model_name: str, path: str):
        """
        Save trained model to file
        
        Args:
            model: Trained model
            model_name: Name of the model
            path: Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_info = {
            'model': model,
            'model_name': model_name,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        joblib.dump(model_info, path)
        print(f"\nModel saved to: {path}")
    
    def load_model(self, path: str):
        """
        Load trained model from file
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded model
        """
        model_info = joblib.load(path)
        self.best_model = model_info['model']
        self.best_model_name = model_info['model_name']
        print(f"Model loaded: {self.best_model_name}")
        return self.best_model


def main():
    """Main function to run model training"""
    print("="*60)
    print("FAKE NEWS DETECTION - MODEL TRAINING")
    print("="*60)
    
    # Load preprocessed data
    print("\nLoading preprocessed data...")
    df = pd.read_csv('data/processed/cleaned_data.csv')
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Initialize model trainer
    trainer = ModelTrainer()
    
    # Prepare data
    trainer.prepare_data(df, engineer, test_size=0.2)
    
    # Save vectorizer and scaler
    engineer.save_vectorizer('models/tfidf_vectorizer.pkl')
    engineer.save_scaler('models/scaler.pkl')
    
    # Train all models
    results = trainer.train_all_models()
    
    # Optional: Tune best model
    print("\nDo you want to perform hyperparameter tuning? (This may take time)")
    print("Skipping tuning for now...")
    
    # Save best model
    trainer.save_model(
        trainer.best_model,
        trainer.best_model_name,
        'models/fake_news_model.pkl'
    )
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETED!")
    print("="*60)
    print(f"\nBest Model: {trainer.best_model_name}")
    print(f"Model saved to: models/fake_news_model.pkl")


if __name__ == "__main__":
    main()