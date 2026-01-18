"""
Batch Prediction Page - Upload CSV and predict multiple articles at once
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import joblib
from io import StringIO

st.set_page_config(page_title="Batch Prediction", page_icon="📁", layout="wide")

st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer

st.title("📁 Batch News Prediction")
st.markdown("### Upload a CSV file to analyze multiple articles at once")

@st.cache_resource
def load_model():
    """Load model and vectorizer"""
    try:
        model_info = joblib.load('models/fake_news_model.pkl')
        model = model_info['model']
        
        engineer = FeatureEngineer()
        engineer.load_vectorizer('models/tfidf_vectorizer.pkl')
        engineer.load_scaler('models/scaler.pkl')
        
        return model, engineer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def predict_batch(texts, model, engineer):
    """Predict multiple articles"""
    preprocessor = DataPreprocessor()
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, text in enumerate(texts):
        try:
            # Preprocess
            cleaned = preprocessor.clean_text(text)
            processed = preprocessor.tokenize_and_lemmatize(cleaned)
            features = preprocessor.extract_features(processed, text)
            
            # Create DataFrame
            df = pd.DataFrame({
                'processed_content': [processed],
                **{k: [v] for k, v in features.items()}
            })
            
            # Predict
            X = engineer.engineer_features(df, fit=False)
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0]
            
            results.append({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'prediction': 'REAL' if prediction == 1 else 'FAKE',
                'confidence': probability[prediction] * 100,
                'fake_prob': probability[0] * 100,
                'real_prob': probability[1] * 100
            })
            
        except Exception as e:
            results.append({
                'text': text[:100] + '...',
                'prediction': 'ERROR',
                'confidence': 0,
                'fake_prob': 0,
                'real_prob': 0
            })
        
        # Update progress
        progress_bar.progress((idx + 1) / len(texts))
        status_text.text(f"Processing {idx + 1}/{len(texts)}...")
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(results)

# Load model
model, engineer = load_model()

if model is None:
    st.error("⚠️ Model not loaded!")
    st.stop()

# Instructions
with st.expander("📋 Instructions"):
    st.markdown("""
    ### How to use:
    
    1. **Prepare your CSV file** with a column containing news articles
    2. **Upload the file** using the file uploader below
    3. **Select the column** that contains the article text
    4. **Click Predict** to analyze all articles
    5. **Download results** as a new CSV file
    
    ### CSV Format:
Your CSV should have a column named **'article'** containing the news text. That's all you need!

**Example:**

| article |
|---------|
| Breaking news: Scientists discover... |
| Federal Reserve announces... |
| You won't believe what happened... |

The tool will add prediction columns (prediction, confidence, fake_prob, real_prob) to your data.

**Note:** You can include other columns (like id, source, date) if you want, but only the 'article' column is required.
    """)

# File upload
uploaded_file = st.file_uploader("📤 Upload CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ File uploaded: {len(df)} rows")
        
        # Show preview
        st.markdown("### 👀 Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        # Select column
        st.markdown("### ⚙️ Configuration")
        text_column = st.selectbox(
            "Select the column containing article text:",
            options=df.columns.tolist()
        )
        
        # Predict button
        if st.button("🔍 Predict All Articles", type="primary"):
            with st.spinner("Analyzing articles..."):
                texts = df[text_column].astype(str).tolist()
                results_df = predict_batch(texts, model, engineer)
                
                # Combine with original data
                final_df = pd.concat([df, results_df[['prediction', 'confidence', 'fake_prob', 'real_prob']]], axis=1)
                
                # Display results
                st.markdown("### 📊 Results")
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Articles", len(final_df))
                
                with col2:
                    fake_count = (final_df['prediction'] == 'FAKE').sum()
                    st.metric("Fake News", fake_count, delta=f"{fake_count/len(final_df)*100:.1f}%")
                
                with col3:
                    real_count = (final_df['prediction'] == 'REAL').sum()
                    st.metric("Real News", real_count, delta=f"{real_count/len(final_df)*100:.1f}%")
                
                with col4:
                    avg_confidence = final_df['confidence'].mean()
                    st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                
                # Distribution chart
                st.markdown("### 📈 Distribution")
                prediction_counts = final_df['prediction'].value_counts()
                st.bar_chart(prediction_counts)
                
                # Results table
                st.markdown("### 📋 Detailed Results")
                st.dataframe(
                    final_df.style.applymap(
                        lambda x: 'background-color: #ffebee' if x == 'FAKE' else ('background-color: #e8f5e9' if x == 'REAL' else ''),
                        subset=['prediction']
                    ),
                    use_container_width=True
                )
                
                # Download results
                st.markdown("### 💾 Download Results")
                
                csv = final_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV with Predictions",
                    data=csv,
                    file_name="fake_news_predictions.csv",
                    mime="text/csv"
                )
                
    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("👆 Upload a CSV file to get started!")
    
    # Sample CSV
st.markdown("### 📄 Sample CSV Template")

sample_data = {
    'article': [
        'Breaking news: Scientists discover amazing breakthrough!',
        'Federal Reserve maintains interest rates unchanged.',
        'You won\'t believe what happened next! Doctors hate this trick!'
    ]
}

sample_df = pd.DataFrame(sample_data)
st.dataframe(sample_df, use_container_width=True)

st.info("💡 Your CSV only needs one column named 'article' with the news text. No ID or source needed!")

csv = sample_df.to_csv(index=False)
st.download_button(
    label="📥 Download Sample CSV",
    data=csv,
    file_name="sample_template.csv",
    mime="text/csv"
)