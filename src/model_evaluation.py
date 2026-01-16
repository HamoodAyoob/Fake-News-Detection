"""
Model Evaluation Module for Fake News Detection
Evaluates model performance and creates visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve
)
from wordcloud import WordCloud
import joblib
import os
from datetime import datetime


class ModelEvaluator:
    """Class to handle model evaluation operations"""
    
    def __init__(self, model, X_test, y_test):
        """
        Initialize evaluator
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = None
        self.y_pred_proba = None
        
    def predict(self):
        """Generate predictions"""
        print("Generating predictions...")
        self.y_pred = self.model.predict(self.X_test)
        
        if hasattr(self.model, 'predict_proba'):
            self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        else:
            self.y_pred_proba = self.model.decision_function(self.X_test)
    
    def calculate_metrics(self) -> dict:
        """
        Calculate all evaluation metrics
        
        Returns:
            Dictionary of metrics
        """
        print("\nCalculating metrics...")
        
        if self.y_pred is None:
            self.predict()
        
        metrics = {
            'accuracy': accuracy_score(self.y_test, self.y_pred),
            'precision_fake': precision_score(self.y_test, self.y_pred, pos_label=0),
            'precision_real': precision_score(self.y_test, self.y_pred, pos_label=1),
            'recall_fake': recall_score(self.y_test, self.y_pred, pos_label=0),
            'recall_real': recall_score(self.y_test, self.y_pred, pos_label=1),
            'f1_macro': f1_score(self.y_test, self.y_pred, average='macro'),
            'f1_weighted': f1_score(self.y_test, self.y_pred, average='weighted'),
            'roc_auc': roc_auc_score(self.y_test, self.y_pred_proba)
        }
        
        return metrics
    
    def print_metrics(self, metrics: dict):
        """
        Print metrics in formatted way
        
        Args:
            metrics: Dictionary of metrics
        """
        print("\n" + "="*60)
        print("MODEL PERFORMANCE METRICS")
        print("="*60)
        
        print(f"\nAccuracy Score: {metrics['accuracy']:.4f}")
        print(f"\nPrecision:")
        print(f"  - Fake News: {metrics['precision_fake']:.4f}")
        print(f"  - Real News: {metrics['precision_real']:.4f}")
        print(f"\nRecall:")
        print(f"  - Fake News: {metrics['recall_fake']:.4f}")
        print(f"  - Real News: {metrics['recall_real']:.4f}")
        print(f"\nF1-Score:")
        print(f"  - Macro: {metrics['f1_macro']:.4f}")
        print(f"  - Weighted: {metrics['f1_weighted']:.4f}")
        print(f"\nROC-AUC Score: {metrics['roc_auc']:.4f}")
        
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(
            self.y_test, self.y_pred,
            target_names=['Fake', 'Real']
        ))
    
    def plot_confusion_matrix(self, save_path: str = None):
        """
        Plot confusion matrix
        
        Args:
            save_path: Path to save the plot
        """
        if self.y_pred is None:
            self.predict()
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fake', 'Real'],
            yticklabels=['Fake', 'Real']
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
            
            # Verify file was created
            if os.path.exists(save_path):
                print(f"  ✅ File exists: {os.path.getsize(save_path)} bytes")
            else:
                print(f"  ❌ File NOT created!")
        
        plt.close()
    
    def plot_roc_curve(self, save_path: str = None):
        """
        Plot ROC curve
        
        Args:
            save_path: Path to save the plot
        """
        if self.y_pred_proba is None:
            self.predict()
        
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = roc_auc_score(self.y_test, self.y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', 
                 fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to: {save_path}")
        
        plt.close()
    
    def plot_precision_recall_curve(self, save_path: str = None):
        """
        Plot precision-recall curve
        
        Args:
            save_path: Path to save the plot
        """
        if self.y_pred_proba is None:
            self.predict()
        
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-Recall curve saved to: {save_path}")
        
        plt.close()
    
    def plot_feature_importance(self, feature_names: list, top_n: int = 20,
                               save_path: str = None):
        """
        Plot feature importance for tree-based models
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to display
            save_path: Path to save the plot
        """
        if not hasattr(self.model, 'feature_importances_'):
            print("Model does not have feature importances")
            return
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title(f'Top {top_n} Most Important Features', 
                 fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to: {save_path}")
        
        plt.close()
    
    def create_wordclouds(self, df: pd.DataFrame, save_dir: str = None):
        """
        Create word clouds for fake and real news
        
        Args:
            df: DataFrame with processed text
            save_dir: Directory to save word clouds
        """
        print("\nGenerating word clouds...")
        
        # Fake news word cloud
        fake_text = ' '.join(df[df['label'] == 0]['processed_content'].values)
        fake_wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='Reds'
        ).generate(fake_text)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(fake_wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Common Words in FAKE News', 
                 fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/fake_news_wordcloud.png", 
                       dpi=300, bbox_inches='tight')
        
        plt.close()
        
        # Real news word cloud
        real_text = ' '.join(df[df['label'] == 1]['processed_content'].values)
        real_wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='Greens'
        ).generate(real_text)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(real_wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Common Words in REAL News', 
                 fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(f"{save_dir}/real_news_wordcloud.png", 
                       dpi=300, bbox_inches='tight')
            print(f"Word clouds saved to: {save_dir}")
        
        plt.close()
    
    def save_metrics_to_file(self, metrics: dict, model_name: str, 
                            file_path: str):
        """
        Save metrics to text file
        
        Args:
            metrics: Dictionary of metrics
            model_name: Name of the model
            file_path: Path to save metrics
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("FAKE NEWS DETECTION - MODEL EVALUATION\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Model: {model_name}\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("="*60 + "\n")
            f.write("PERFORMANCE METRICS\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n\n")
            
            f.write("Precision:\n")
            f.write(f"  Fake News: {metrics['precision_fake']:.4f}\n")
            f.write(f"  Real News: {metrics['precision_real']:.4f}\n\n")
            
            f.write("Recall:\n")
            f.write(f"  Fake News: {metrics['recall_fake']:.4f}\n")
            f.write(f"  Real News: {metrics['recall_real']:.4f}\n\n")
            
            f.write("F1-Score:\n")
            f.write(f"  Macro: {metrics['f1_macro']:.4f}\n")
            f.write(f"  Weighted: {metrics['f1_weighted']:.4f}\n\n")
            
            f.write(f"ROC-AUC Score: {metrics['roc_auc']:.4f}\n\n")
            
            f.write("="*60 + "\n")
            f.write("CLASSIFICATION REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(classification_report(
                self.y_test, self.y_pred,
                target_names=['Fake', 'Real']
            ))
        
        print(f"\nMetrics saved to: {file_path}")


def main():
    """Main function to run model evaluation"""
    print("="*60)
    print("FAKE NEWS DETECTION - MODEL EVALUATION")
    print("="*60)
    
    # Load model
    print("\nLoading trained model...")
    model_info = joblib.load('models/fake_news_model.pkl')
    model = model_info['model']
    model_name = model_info['model_name']
    
    print(f"Loaded model: {model_name}")
    
    # Load test data
    from model_training import ModelTrainer
    from feature_engineering import FeatureEngineer
    
    df = pd.read_csv('data/processed/cleaned_data.csv')
    engineer = FeatureEngineer()
    engineer.load_vectorizer('models/tfidf_vectorizer.pkl')
    engineer.load_scaler('models/scaler.pkl')
    
    trainer = ModelTrainer()
    trainer.best_model = model
    X_train, X_test, y_train, y_test = trainer.prepare_data(df, engineer)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model, X_test, y_test)
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics()
    evaluator.print_metrics(metrics)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    print("  1. Confusion Matrix...")
    evaluator.plot_confusion_matrix('models/confusion_matrix.png')
    
    print("  2. ROC Curve...")
    evaluator.plot_roc_curve('models/roc_curve.png')
    
    print("  3. Precision-Recall Curve...")
    evaluator.plot_precision_recall_curve('models/precision_recall_curve.png')
    
    # Feature importance (if applicable)
    if hasattr(model, 'feature_importances_'):
        print("  4. Feature Importance...")
        feature_names = engineer.get_feature_names()
        evaluator.plot_feature_importance(
            feature_names, 
            top_n=20, 
            save_path='models/feature_importance.png'
        )
    else:
        print("  4. Feature Importance... (skipped - not available for this model)")
    
    # Word clouds
    print("  5. Word Clouds...")
    evaluator.create_wordclouds(df, save_dir='models')
    
    # Save metrics
    print("\nSaving metrics to file...")
    evaluator.save_metrics_to_file(metrics, model_name, 'models/model_metrics.txt')
    
    # List all generated files
    print("\n" + "="*60)
    print("GENERATED FILES")
    print("="*60)
    
    files_to_check = [
        'models/model_metrics.txt',
        'models/confusion_matrix.png',
        'models/roc_curve.png',
        'models/precision_recall_curve.png',
        'models/feature_importance.png',
        'models/fake_news_wordcloud.png',
        'models/real_news_wordcloud.png'
    ]
    
    for file in files_to_check:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✅ {file} ({size:,} bytes)")
        else:
            print(f"❌ {file} (not created)")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETED!")
    print("="*60)
    print("\nView visualizations in the 'models/' folder")
    print("Or run the Streamlit app to see them:")
    print("  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()