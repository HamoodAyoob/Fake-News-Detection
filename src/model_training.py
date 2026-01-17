"""
Retrain Model with Better Generalization
Fix overfitting issue
"""

import pandas as pd
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
from datetime import datetime

sys.path.append('src')
from feature_engineering import FeatureEngineer

print("="*60)
print("RETRAINING MODEL - IMPROVED VERSION")
print("="*60)

# Load data
print("\nLoading data...")
df = pd.read_csv('data/processed/cleaned_data.csv')
print(f"✅ Loaded {len(df)} articles")

# Engineer features
print("\nEngineering features...")
engineer = FeatureEngineer()

# Reduce TF-IDF features to prevent overfitting
engineer.tfidf_vectorizer.max_features = 3000  # Reduced from 5000
engineer.tfidf_vectorizer.min_df = 5  # Increased from 2
engineer.tfidf_vectorizer.max_df = 0.7  # Reduced from 0.8

X = engineer.engineer_features(df, fit=True)
y = df['label'].values.astype(int)

print(f"✅ Feature shape: {X.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")

# Train models with better regularization
print("\n" + "="*60)
print("TRAINING MODELS")
print("="*60)

models = {
    'Logistic Regression': LogisticRegression(
        C=0.1,  # Strong regularization
        max_iter=1000,
        random_state=42,
        solver='saga'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,  # Reduced from 200
        max_depth=20,  # Reduced from 30
        min_samples_split=10,  # Increased from 5
        min_samples_leaf=5,  # Added
        max_features='sqrt',  # Changed from auto
        random_state=42,
        n_jobs=-1
    )
}

results = {}

for name, model in models.items():
    print(f"\n{name}:")
    print("-"*40)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"CV Scores: {cv_scores}")
    print(f"Mean CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Train
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Train Accuracy: {train_score:.4f}")
    print(f"Test Accuracy: {test_score:.4f}")
    
    # Check for overfitting
    overfitting = train_score - test_score
    print(f"Overfitting gap: {overfitting:.4f}")
    
    if overfitting > 0.05:
        print("⚠️ Warning: Model may be overfitted")
    else:
        print("✅ Good generalization")
    
    results[name] = {
        'model': model,
        'cv_mean': cv_scores.mean(),
        'test_score': test_score,
        'overfitting': overfitting
    }

# Select best model (lowest overfitting, good accuracy)
print("\n" + "="*60)
print("SELECTING BEST MODEL")
print("="*60)

# Prefer model with less overfitting
best_name = min(results.keys(), key=lambda k: results[k]['overfitting'])
best_model = results[best_name]['model']

print(f"\nBest Model: {best_name}")
print(f"Test Accuracy: {results[best_name]['test_score']:.4f}")
print(f"Overfitting: {results[best_name]['overfitting']:.4f}")

# Save model
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

model_info = {
    'model': best_model,
    'model_name': best_name,
    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'version': '2.0 - Improved Generalization'
}

joblib.dump(model_info, 'models/fake_news_model.pkl')
print("✅ Model saved: models/fake_news_model.pkl")

# Save vectorizer
engineer.save_vectorizer('models/tfidf_vectorizer.pkl')
engineer.save_scaler('models/scaler.pkl')

print("\n" + "="*60)
print("TESTING ON REAL NEWS SAMPLES")
print("="*60)

# Quick test
from data_preprocessing import DataPreprocessor

test_article = """The Federal Reserve announced that it will maintain interest rates 
at current levels. The decision comes after careful analysis of economic indicators 
and inflation data. Fed Chairman stated that the economy shows signs of steady growth."""

preprocessor = DataPreprocessor()
cleaned = preprocessor.clean_text(test_article)
processed = preprocessor.tokenize_and_lemmatize(cleaned)
features = preprocessor.extract_features(processed, test_article)

test_df = pd.DataFrame({
    'processed_content': [processed],
    **{k: [v] for k, v in features.items()}
})

X_test_sample = engineer.engineer_features(test_df, fit=False)
prediction = best_model.predict(X_test_sample)[0]
probability = best_model.predict_proba(X_test_sample)[0]

print("\nTest Article (Real News):")
print(test_article[:100] + "...")
print(f"\nPrediction: {'FAKE' if prediction == 0 else 'REAL'}")
print(f"Confidence: {probability[prediction]*100:.2f}%")
print(f"Fake: {probability[0]*100:.2f}% | Real: {probability[1]*100:.2f}%")

if prediction == 1:
    print("\n✅ SUCCESS! Model correctly identifies real news!")
else:
    print("\n⚠️ Still having issues. May need more diverse training data.")

print("\n" + "="*60)
print("RETRAINING COMPLETE!")
print("="*60)
print("\nNext steps:")
print("1. Test the new model: python test_real_articles.py")
print("2. Launch app: streamlit run app/streamlit_app.py")