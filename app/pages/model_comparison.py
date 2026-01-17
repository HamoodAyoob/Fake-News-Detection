"""
Model Comparison Page - Compare all trained models
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Model Comparison", page_icon="📊", layout="wide")

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

st.title("📊 Model Performance Comparison")
st.markdown("### Compare all 5 trained models")

# Sample data - UPDATE THESE with your actual training results
model_data = {
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost', 'SVM', 'Naive Bayes'],
    'Accuracy': [95.20, 92.30, 91.80, 89.70, 86.20],  # Updated with realistic values
    'Precision': [94.85, 91.10, 90.45, 88.75, 85.40],
    'Recall': [95.10, 92.15, 91.40, 89.70, 86.45],
    'F1-Score': [95.05, 91.62, 90.92, 89.22, 85.92],
    'Training Time (s)': [45, 480, 320, 180, 15]
}

df = pd.DataFrame(model_data)

# Display metrics table
st.markdown("### 📈 Performance Metrics")
st.dataframe(
    df.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'], color='lightgreen')
    .highlight_min(subset=['Training Time (s)'], color='lightblue')
    .format({'Accuracy': '{:.2f}%', 'Precision': '{:.2f}%', 'Recall': '{:.2f}%', 'F1-Score': '{:.2f}%'}),
    use_container_width=True
)

# Bar chart comparison
st.markdown("### 📊 Accuracy Comparison")
fig = px.bar(df, x='Model', y='Accuracy', 
             color='Accuracy',
             color_continuous_scale='Viridis',
             text='Accuracy')
fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)

# Multi-metric comparison
st.markdown("### 📉 Multi-Metric Comparison")
fig = go.Figure()

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
for metric in metrics:
    fig.add_trace(go.Bar(name=metric, x=df['Model'], y=df[metric]))

fig.update_layout(barmode='group', height=500)
st.plotly_chart(fig, use_container_width=True)

# Training time comparison
st.markdown("### ⏱️ Training Time Comparison")
fig = px.bar(df, x='Model', y='Training Time (s)',
             color='Training Time (s)',
             color_continuous_scale='Reds_r',
             text='Training Time (s)')
fig.update_traces(texttemplate='%{text:.0f}s', textposition='outside')
fig.update_layout(height=400)
st.plotly_chart(fig, use_container_width=True)

# Best model recommendation
st.markdown("### 🏆 Recommendation")
best_model = df.loc[df['Accuracy'].idxmax()]
st.success(f"""
**Best Model: {best_model['Model']}**
- Accuracy: {best_model['Accuracy']:.2f}%
- Precision: {best_model['Precision']:.2f}%
- Recall: {best_model['Recall']:.2f}%
- F1-Score: {best_model['F1-Score']:.2f}%
- Training Time: {best_model['Training Time (s)']:.0f}s
""")

# Model characteristics
st.markdown("### 🔍 Model Characteristics")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**🌳 Random Forest**")
    st.info("""
    - Ensemble of decision trees
    - Very high accuracy
    - Good with imbalanced data
    - Slower training time
    - Less prone to overfitting
    """)
    
    st.markdown("**🚀 XGBoost**")
    st.info("""
    - Gradient boosting algorithm
    - Excellent performance
    - Fast training
    - Handles missing data well
    - Good generalization
    """)
    
    st.markdown("**📈 Logistic Regression**")
    st.info("""
    - Linear model
    - Very fast training
    - Good baseline
    - Interpretable coefficients
    - Works well for text
    """)

with col2:
    st.markdown("**🎯 SVM**")
    st.info("""
    - Support Vector Machine
    - Good with high dimensions
    - Effective in text classification
    - Moderate training time
    - Robust to outliers
    """)
    
    st.markdown("**🧮 Naive Bayes**")
    st.info("""
    - Probabilistic classifier
    - Fastest training
    - Good for text data
    - Simple and efficient
    - Assumes feature independence
    """)