"""
Streamlit Frontend for Fake News Detection System
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import joblib
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer


# Page configuration
st.set_page_config(
    page_title="🔍 Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    """Load custom CSS file"""
    css_file = os.path.join(os.path.dirname(__file__), 'assets', 'style.css')
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        # Fallback inline CSS if file not found
        st.markdown("""
            <style>
            .main {
                padding: 0rem 1rem;
            }
            .fake-news {
                background-color: #ffebee;
                color: #c62828;
                padding: 2rem;
                border-radius: 1rem;
                border-left: 5px solid #c62828;
                margin: 1rem 0;
            }
            .real-news {
                background-color: #e8f5e9;
                color: #2e7d32;
                padding: 2rem;
                border-radius: 1rem;
                border-left: 5px solid #2e7d32;
                margin: 1rem 0;
            }
            h1 {
                color: #1976d2;
            }
            </style>
        """, unsafe_allow_html=True)

# Load CSS
load_css()


@st.cache_resource
def load_model_and_vectorizer():
    """Load trained model and vectorizer"""
    try:
        model_info = joblib.load('models/fake_news_model.pkl')
        model = model_info['model']
        model_name = model_info['model_name']
        
        engineer = FeatureEngineer()
        engineer.load_vectorizer('models/tfidf_vectorizer.pkl')
        engineer.load_scaler('models/scaler.pkl')
        
        return model, model_name, engineer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


def preprocess_input(text: str, engineer: FeatureEngineer) -> np.ndarray:
    """Preprocess user input text"""
    preprocessor = DataPreprocessor()
    
    # Clean text
    cleaned = preprocessor.clean_text(text)
    
    # Tokenize and lemmatize
    processed = preprocessor.tokenize_and_lemmatize(cleaned)
    
    # Extract features
    features = preprocessor.extract_features(processed, text)
    
    # Create DataFrame for feature engineering
    df = pd.DataFrame({
        'processed_content': [processed],
        **{k: [v] for k, v in features.items()}
    })
    
    # Engineer features
    final_features = engineer.engineer_features(df, fit=False)
    
    return final_features


def predict_news(text: str, model, engineer: FeatureEngineer):
    """Make prediction on input text"""
    # Preprocess
    features = preprocess_input(text, engineer)
    
    # Predict
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    return prediction, probability


def home_page():
    """Home page with prediction interface"""
    st.title("🔍 Fake News Detector")
    st.markdown("### Analyze news articles using Machine Learning")
    
    # Load model
    model, model_name, engineer = load_model_and_vectorizer()
    
    if model is None:
        st.error("⚠️ Model not found! Please train the model first.")
        st.info("Run: `python run_project.py` and select option 3 to train the model.")
        return
    
    st.success(f"✅ Model loaded: **{model_name}**")
    
    # Input section
    st.markdown("---")
    st.markdown("### 📝 Enter News Article")
    
    # Sample texts
    with st.expander("📋 Use Sample Text"):
        col1, col2 = st.columns(2)
        
        sample_fake = """
        BREAKING: Scientists discover that drinking coffee cures all diseases! 
        A new study shows that coffee can cure cancer, diabetes, and even aging. 
        Doctors are shocked by this discovery. Share this with everyone you know!
        """
        
        sample_real = """
        The Federal Reserve announced today that it will maintain interest rates 
        at current levels. The decision comes after careful analysis of economic 
        indicators and inflation data. Fed Chairman stated that the economy 
        shows signs of steady growth.
        """
        
        if col1.button("Load Fake News Sample"):
            st.session_state['input_text'] = sample_fake
        
        if col2.button("Load Real News Sample"):
            st.session_state['input_text'] = sample_real
    
    # Text input
    input_text = st.text_area(
        "Paste or type the news article here:",
        value=st.session_state.get('input_text', ''),
        height=200,
        placeholder="Enter the news article text here..."
    )
    
    # Analyze button
    col1, col2, col3 = st.columns([1, 1, 3])
    
    with col1:
        analyze_button = st.button("🔍 Analyze News", type="primary", use_container_width=True)
    
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state['input_text'] = ''
            st.rerun()
    
    # Prediction
    if analyze_button:
        if not input_text.strip():
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            with st.spinner("🔄 Analyzing..."):
                try:
                    prediction, probability = predict_news(input_text, model, engineer)
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("### 📊 Analysis Results")
                    
                    if prediction == 0:
                        st.markdown(f"""
                        <div class="fake-news">
                            <h2>❌ FAKE NEWS</h2>
                            <p style="font-size: 1.2em;">
                                This article appears to be <strong>FAKE</strong> or misleading.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="real-news">
                            <h2>✅ REAL NEWS</h2>
                            <p style="font-size: 1.2em;">
                                This article appears to be <strong>REAL</strong> and credible.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Confidence scores
                    st.markdown("### 📈 Confidence Scores")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Fake News Probability",
                            f"{probability[0]*100:.2f}%"
                        )
                    
                    with col2:
                        st.metric(
                            "Real News Probability",
                            f"{probability[1]*100:.2f}%"
                        )
                    
                    # Probability chart
                    fig = go.Figure(data=[
                        go.Bar(
                            x=['Fake', 'Real'],
                            y=[probability[0]*100, probability[1]*100],
                            marker_color=['#ef5350', '#66bb6a'],
                            text=[f"{probability[0]*100:.2f}%", f"{probability[1]*100:.2f}%"],
                            textposition='auto'
                        )
                    ])
                    
                    fig.update_layout(
                        title="Prediction Probability",
                        yaxis_title="Probability (%)",
                        xaxis_title="Category",
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {e}")
    
    # Tips section
    st.markdown("---")
    with st.expander("💡 Tips for Using the Detector"):
        st.markdown("""
        - **Provide complete articles**: The model works best with full article text
        - **Include context**: Headlines alone may not provide enough information
        - **Multiple sources**: Cross-check with other reliable news sources
        - **Critical thinking**: Use this tool as one of many fact-checking methods
        - **Limitations**: No AI is perfect - always verify important information
        """)


def performance_page():
    """Model performance page"""
    st.title("📊 Model Performance")
    st.markdown("### Detailed evaluation metrics and visualizations")
    
    # Check if metrics file exists
    if not os.path.exists('models/model_metrics.txt'):
        st.warning("⚠️ Model metrics not found. Please evaluate the model first.")
        st.info("Run: `python run_project.py` and select option 4")
        return
    
    # Load and display metrics
    with open('models/model_metrics.txt', 'r') as f:
        metrics_text = f.read()
    
    st.markdown("### 📈 Performance Metrics")
    st.text(metrics_text)
    
    st.markdown("---")
    
    # Display visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Confusion Matrix")
        if os.path.exists('models/confusion_matrix.png'):
            st.image('models/confusion_matrix.png')
        else:
            st.info("Confusion matrix not available")
    
    with col2:
        st.markdown("#### ROC Curve")
        if os.path.exists('models/roc_curve.png'):
            st.image('models/roc_curve.png')
        else:
            st.info("ROC curve not available")
    
    st.markdown("---")
    
    # Precision-Recall curve
    st.markdown("#### Precision-Recall Curve")
    if os.path.exists('models/precision_recall_curve.png'):
        st.image('models/precision_recall_curve.png', use_container_width=True)
    
    st.markdown("---")
    
    # Feature importance
    st.markdown("#### Feature Importance")
    if os.path.exists('models/feature_importance.png'):
        st.image('models/feature_importance.png', use_container_width=True)
    else:
        st.info("Feature importance not available (only for tree-based models)")
    
    st.markdown("---")
    
    # Word clouds
    st.markdown("### ☁️ Word Clouds")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Fake News")
        if os.path.exists('models/fake_news_wordcloud.png'):
            st.image('models/fake_news_wordcloud.png')
    
    with col2:
        st.markdown("#### Real News")
        if os.path.exists('models/real_news_wordcloud.png'):
            st.image('models/real_news_wordcloud.png')


def about_page():
    """About page with project information"""
    st.title("ℹ️ About This Project")
    
    st.markdown("""
    ### 🎯 Project Overview
    
    This **Fake News Detection System** uses Machine Learning and Natural Language Processing 
    to classify news articles as either REAL or FAKE. The system analyzes text content using 
    various NLP techniques and statistical features to make predictions.
    
    ---
    
    ### 📊 Dataset Information
    
    - **Source**: Kaggle Fake and Real News Dataset
    - **Size**: ~45,000 labeled articles
    - **Classes**: Fake (0) and Real (1)
    - **Features**: Title, Text, Subject, Date
    
    ---
    
    ### 🤖 Model Details
    
    **Text Preprocessing:**
    - Text cleaning (URLs, special characters, etc.)
    - Tokenization using NLTK
    - Stopword removal
    - Lemmatization
    
    **Feature Engineering:**
    - TF-IDF vectorization (5000 features, unigrams & bigrams)
    - Statistical features (text length, word count, sentiment, etc.)
    
    **Models Trained:**
    - Logistic Regression
    - Naive Bayes
    - Support Vector Machine (SVM)
    - Random Forest
    - XGBoost
    
    **Best Model Selection:**
    - Models compared based on test accuracy
    - Cross-validation for robustness
    - Comprehensive evaluation metrics
    
    ---
    
    ### 📈 Performance Expectations
    
    - **Accuracy**: >90%
    - **Precision & Recall**: >85% for both classes
    - **F1-Score**: >88%
    - **Response Time**: <2 seconds
    
    ---
    
    ### 🚀 How to Use
    
    1. **Home Page**: Enter a news article in the text area
    2. **Click Analyze**: The system will classify the article
    3. **Review Results**: Check the prediction and confidence scores
    4. **Performance**: View detailed model metrics
    
    ---
    
    ### ⚠️ Limitations
    
    - The model is trained on specific datasets and may not generalize to all types of news
    - Context and sarcasm can be difficult to detect
    - Always cross-verify important information with multiple sources
    - No AI system is 100% accurate
    
    ---
    
    ### 🛠️ Technology Stack
    
    - **Language**: Python 3.9+
    - **ML Libraries**: scikit-learn, XGBoost
    - **NLP**: NLTK, TextBlob
    - **Visualization**: Matplotlib, Seaborn, Plotly
    - **Web Framework**: Streamlit
    
    ---
    
    ### 📝 Future Improvements
    
    - Deep learning models (LSTM, BERT)
    - Multi-language support
    - Real-time news fetching
    - User feedback mechanism
    - Source credibility checker
    
    ---
    
    ### 👥 Credits
    
    - **Dataset**: Clément Bisaillon (Kaggle)
    - **Libraries**: scikit-learn, NLTK, Streamlit, and open-source community
    
    ---
    
    ### 📄 License
    
    This project is for educational purposes. Please ensure proper attribution 
    when using or modifying this code.
    """)


def main():
    """Main function"""
    # Sidebar navigation
    st.sidebar.title("🔍 Navigation")
    
    page = st.sidebar.radio(
        "Go to:",
        ["🏠 Home", "📊 Model Performance", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Quick Stats")
    
    if os.path.exists('models/fake_news_model.pkl'):
        model_info = joblib.load('models/fake_news_model.pkl')
        st.sidebar.success(f"**Model**: {model_info['model_name']}")
        st.sidebar.info(f"**Trained**: {model_info['timestamp']}")
    else:
        st.sidebar.warning("Model not trained yet")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 💡 Tips
    - Enter complete articles for best results
    - Check multiple sources
    - Use critical thinking
    """)
    
    # Route to pages
    if page == "🏠 Home":
        home_page()
    elif page == "📊 Model Performance":
        performance_page()
    else:
        about_page()


if __name__ == "__main__":
    main()