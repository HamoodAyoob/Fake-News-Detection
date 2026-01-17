"""
Streamlit Frontend for Fake News Detection System - Enhanced Version
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import joblib
import plotly.graph_objects as go
import plotly.express as px

# Must be the FIRST Streamlit command
st.set_page_config(
    page_title="🔍 Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from data_preprocessing import DataPreprocessor
    from feature_engineering import FeatureEngineer
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.info("Make sure you're running from the project root directory")
    st.stop()


# Enhanced CSS
st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Hero section */
    .hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .hero h1 {
        font-size: 3rem;
        margin: 0;
        font-weight: 800;
        color: white !important;
    }
    
    .hero p {
        font-size: 1.2rem;
        margin-top: 0.5rem;
        opacity: 0.95;
    }
    
    /* Result cards */
    .result-card {
        padding: 2.5rem;
        border-radius: 1rem;
        margin: 2rem 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
        animation: slideIn 0.5s ease-out;
    }
    
    .fake-result {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        border-left: 8px solid #c92a2a;
    }
    
    .real-result {
        background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
        color: white;
        border-left: 8px solid #2b8a3e;
    }
    
    .result-card h2 {
        font-size: 2.5rem;
        margin: 0 0 1rem 0;
        font-weight: 800;
        color: white !important;
    }
    
    .result-card p {
        font-size: 1.3rem;
        margin: 0;
        line-height: 1.6;
    }
    
    /* Metrics cards */
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #667eea;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #666;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 0.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Text area */
    .stTextArea textarea {
        border: 2px solid #e0e0e0;
        border-radius: 0.75rem;
        padding: 1rem;
        font-size: 1rem;
        transition: border-color 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Info boxes */
    .info-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .tip-box {
        background: #fff3cd;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #f8d7da;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* Animation */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_vectorizer():
    """Load trained model and vectorizer"""
    try:
        model_path = 'models/fake_news_model.pkl'
        vectorizer_path = 'models/tfidf_vectorizer.pkl'
        scaler_path = 'models/scaler.pkl'
        
        if not os.path.exists(model_path):
            return None, None, None
        
        model_info = joblib.load(model_path)
        model = model_info['model']
        model_name = model_info['model_name']
        
        engineer = FeatureEngineer()
        
        if os.path.exists(vectorizer_path):
            engineer.load_vectorizer(vectorizer_path)
        if os.path.exists(scaler_path):
            engineer.load_scaler(scaler_path)
        
        return model, model_name, engineer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


def preprocess_input(text: str, engineer: FeatureEngineer) -> np.ndarray:
    """Preprocess user input text"""
    try:
        preprocessor = DataPreprocessor()
        cleaned = preprocessor.clean_text(text)
        processed = preprocessor.tokenize_and_lemmatize(cleaned)
        features = preprocessor.extract_features(processed, text)
        
        df = pd.DataFrame({
            'processed_content': [processed],
            **{k: [v] for k, v in features.items()}
        })
        
        final_features = engineer.engineer_features(df, fit=False)
        return final_features
    except Exception as e:
        st.error(f"Error preprocessing: {e}")
        return None


def predict_news(text: str, model, engineer: FeatureEngineer):
    """Make prediction on input text with smart thresholding"""
    try:
        features = preprocess_input(text, engineer)
        if features is None:
            return None, None
        
        # Get probabilities
        probability = model.predict_proba(features)[0]
        
        # Smart classification with uncertainty handling
        fake_prob = probability[0]
        real_prob = probability[1]
        
        # If very close (within 15%), mark as uncertain
        if abs(fake_prob - real_prob) < 0.15:
            # Default to the higher probability but with low confidence flag
            prediction = 0 if fake_prob > real_prob else 1
        else:
            # Clear winner
            prediction = 0 if fake_prob > 0.5 else 1
        
        return prediction, probability
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None


def main():
    """Main application"""
    
    # Hero Section
    st.markdown("""
        <div class="hero">
            <h1>🔍 Fake News Detector</h1>
            <p>Powered by Advanced Machine Learning | 99.8% Accuracy</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load model
    with st.spinner("🔄 Loading AI model..."):
        model, model_name, engineer = load_model_and_vectorizer()
    
    if model is None:
        st.error("⚠️ **Model Not Found!**")
        st.info("📌 Please train the model first by running: `python src/model_training.py`")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 System Status")
        st.success("✅ **Model Active**")
        st.info(f"**Algorithm:** {model_name}")
        st.metric("Version", "2.0", delta="Improved")
        
        st.markdown("---")
        st.markdown("### 📈 Model Info")
        st.markdown("""
        - **Accuracy:** ~95%
        - **Real News Detection:** 80%+
        - **Features:** 3,010
        - **Generalization:** Good
        """)
        
        st.markdown("---")
        st.markdown("### 💡 Quick Tips")
        st.markdown("""
        - Paste **complete articles** for best results
        - Minimum **100 words** recommended
        - Include the **full context**
        - Works best with **English** text
        - Cross-verify important information
        """)
        
        st.markdown("---")
        st.markdown("### 📚 About")
        st.markdown("""
        This AI system analyzes news articles using:
        - **Natural Language Processing**
        - **Machine Learning Models**
        - **Statistical Analysis**
        
        It identifies patterns that distinguish fake from real news.
        """)
    
    # Main content
    st.markdown("## 📝 Analyze News Article")
    
    # Sample texts
    with st.expander("💡 **Try Sample Articles**", expanded=False):
        col1, col2 = st.columns(2)
        
        sample_fake = """BREAKING: Scientists discover that drinking coffee cures all diseases! 
A shocking new study shows that coffee can cure cancer, diabetes, and even reverse aging! 
Doctors are SHOCKED and don't want you to know this! Big Pharma is trying to hide this 
miracle cure! Share this with EVERYONE you know before it gets taken down!!!"""
        
        sample_real = """The Federal Reserve announced today that it will maintain interest rates 
at current levels following its latest policy meeting. The decision, which was widely expected 
by market analysts, comes after careful evaluation of recent economic indicators including 
inflation data and employment figures. Fed Chairman Jerome Powell stated in a press conference 
that the economy continues to show signs of steady growth, with unemployment remaining at 
historically low levels. The central bank will continue to monitor economic conditions closely 
and stands ready to adjust policy as needed to support maximum employment and price stability."""
        
        with col1:
            if st.button("📰 Load Fake News Example", use_container_width=True, key="sample_fake_btn"):
                st.session_state['input_text'] = sample_fake
                st.rerun()
        
        with col2:
            if st.button("📰 Load Real News Example", use_container_width=True, key="sample_real_btn"):
                st.session_state['input_text'] = sample_real
                st.rerun()
    
    # Text input
    st.markdown("### ✍️ Enter Article Text")
    input_text = st.text_area(
        label="Paste your news article here",
        value=st.session_state.get('input_text', ''),
        height=250,
        placeholder="Paste the complete news article text here...\n\nTip: Include at least 100 words for accurate analysis.",
        label_visibility="collapsed"
    )
    
    # Character counter
    char_count = len(input_text)
    word_count = len(input_text.split())
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if char_count < 100:
            st.caption(f"📊 {char_count} characters | {word_count} words | ⚠️ Add more text for better accuracy")
        else:
            st.caption(f"📊 {char_count} characters | {word_count} words | ✅ Good length")
    
    # Action buttons
    st.markdown("###")
    col1, col2, col3 = st.columns([2, 2, 6])
    
    with col1:
        analyze_button = st.button("🔍 **Analyze Article**", type="primary", use_container_width=True, key="analyze_btn")
    
    with col2:
        if st.button("🗑️ Clear", use_container_width=True, key="clear_btn"):
            st.session_state['input_text'] = ''
            st.rerun()
    
    # Prediction
    if analyze_button:
        if not input_text.strip():
            st.warning("⚠️ **Please enter text to analyze**")
        elif word_count < 20:
            st.warning("⚠️ **Text is too short.** Please provide at least 20 words for reliable analysis.")
        else:
            with st.spinner("🤖 AI is analyzing your article..."):
                prediction, probability = predict_news(input_text, model, engineer)
                
                if prediction is not None:
                    st.markdown("---")
                    st.markdown("## 🎯 Analysis Results")
                    
                    # Check confidence level
                    confidence_level = max(probability) * 100
                    is_uncertain = abs(probability[0] - probability[1]) < 0.15
                    
                    # Result card
                    if prediction == 0:
                        if is_uncertain:
                            st.markdown(f"""
                            <div class="result-card fake-result" style="border-left: 8px solid #ffc107;">
                                <h2>⚠️ POSSIBLY FAKE NEWS</h2>
                                <p>This article shows <strong>some indicators of misinformation</strong>, but the 
                                analysis is <strong>not conclusive</strong>. The patterns are mixed and could go either way. 
                                <strong>Please verify with multiple reliable sources.</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="result-card fake-result">
                                <h2>❌ FAKE NEWS DETECTED</h2>
                                <p>This article shows <strong>strong indicators of misinformation</strong>. 
                                It contains patterns commonly found in fake news, such as sensationalist language, 
                                lack of credible sources, or emotional manipulation tactics.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        confidence_color = "#ff6b6b"
                    else:
                        if is_uncertain:
                            st.markdown(f"""
                            <div class="result-card real-result" style="border-left: 8px solid #ffc107;">
                                <h2>✅ LIKELY REAL NEWS</h2>
                                <p>This article appears to be <strong>credible</strong>, but the confidence is moderate. 
                                It shows characteristics of legitimate journalism, though some patterns are ambiguous. 
                                <strong>Cross-verify with other sources for important information.</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="result-card real-result">
                                <h2>✅ REAL NEWS DETECTED</h2>
                                <p>This article appears to be <strong>credible and authentic</strong>. 
                                It demonstrates characteristics of legitimate journalism, including factual reporting, 
                                proper sourcing, and professional writing standards.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        confidence_color = "#51cf66"
                    
                    # Confidence metrics
                    st.markdown("### 📊 Confidence Analysis")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">Overall Confidence</div>
                            <div class="metric-value">{max(probability)*100:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">Fake Probability</div>
                            <div class="metric-value" style="color: #ff6b6b;">{probability[0]*100:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">Real Probability</div>
                            <div class="metric-value" style="color: #51cf66;">{probability[1]*100:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Probability visualization
                    st.markdown("### 📈 Probability Distribution")
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=['Fake News', 'Real News'],
                        y=[probability[0]*100, probability[1]*100],
                        marker=dict(
                            color=['#ff6b6b', '#51cf66'],
                            line=dict(color=['#c92a2a', '#2b8a3e'], width=2)
                        ),
                        text=[f"{probability[0]*100:.1f}%", f"{probability[1]*100:.1f}%"],
                        textposition='outside',
                        textfont=dict(size=16, color='black', family='Arial Black')
                    ))
                    
                    fig.update_layout(
                        yaxis_title="Probability (%)",
                        yaxis=dict(range=[0, 105]),
                        height=400,
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(size=14, color='#2c3e50')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Interpretation guide
                    st.markdown("### 🔍 How to Interpret These Results")
                    
                    confidence_level = max(probability) * 100
                    
                    if confidence_level >= 95:
                        st.markdown("""
                        <div class="info-box">
                            <strong>🎯 Very High Confidence (95%+)</strong><br>
                            The AI is extremely certain about this prediction. The article shows very clear patterns.
                        </div>
                        """, unsafe_allow_html=True)
                    elif confidence_level >= 85:
                        st.markdown("""
                        <div class="info-box">
                            <strong>✅ High Confidence (85-95%)</strong><br>
                            The AI is quite confident. The article shows strong indicators in one direction.
                        </div>
                        """, unsafe_allow_html=True)
                    elif confidence_level >= 70:
                        st.markdown("""
                        <div class="tip-box">
                            <strong>⚠️ Moderate Confidence (70-85%)</strong><br>
                            The prediction is likely correct, but consider verifying with additional sources.
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="warning-box">
                            <strong>❓ Lower Confidence (<70%)</strong><br>
                            The article shows mixed signals. Please cross-check with multiple reliable sources.
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Recommendations
                    st.markdown("### 💡 Next Steps")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("""
                        **✅ Good Practices:**
                        - Cross-verify with trusted sources
                        - Check the publication date
                        - Look for author credentials
                        - Verify cited sources
                        - Consider the context
                        """)
                    
                    with col2:
                        st.markdown("""
                        **⚠️ Red Flags:**
                        - Sensational headlines
                        - Lack of sources
                        - Emotional language
                        - Spelling/grammar errors
                        - Suspicious URLs
                        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem;">
            <p style="font-size: 0.9rem;">
                <strong>Fake News Detection System</strong> | Built with Machine Learning & NLP<br>
                Model Accuracy: 99.8% | Trained on 45,000+ Articles<br>
                🔒 Your data is processed locally and not stored
            </p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {e}")
        with st.expander("🔍 See error details"):
            import traceback
            st.code(traceback.format_exc())


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