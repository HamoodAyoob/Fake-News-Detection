"""
Dataset Statistics Page - Show insights about the training data
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Dataset Statistics", page_icon="📊", layout="wide")

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

st.title("📊 Dataset Statistics & Insights")
st.markdown("### Explore the training dataset characteristics")

# Load data if available
data_file = 'data/processed/cleaned_data.csv'

if os.path.exists(data_file):
    try:
        df = pd.read_csv(data_file)
        
        # Overview
        st.markdown("## 📈 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Articles", f"{len(df):,}")
        
        with col2:
            fake_count = (df['label'] == 0).sum()
            st.metric("Fake News", f"{fake_count:,}", 
                     delta=f"{fake_count/len(df)*100:.1f}%")
        
        with col3:
            real_count = (df['label'] == 1).sum()
            st.metric("Real News", f"{real_count:,}",
                     delta=f"{real_count/len(df)*100:.1f}%")
        
        with col4:
            balance_ratio = min(fake_count, real_count) / max(fake_count, real_count)
            st.metric("Balance Ratio", f"{balance_ratio:.2f}",
                     help="Ratio of minority to majority class (1.0 = perfect balance)")
        
        # Class distribution
        st.markdown("## 📊 Class Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            labels = ['Fake', 'Real']
            values = [fake_count, real_count]
            
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
            fig.update_traces(marker=dict(colors=['#ef5350', '#66bb6a']))
            fig.update_layout(title='Class Distribution', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Bar chart
            class_df = pd.DataFrame({
                'Class': labels,
                'Count': values
            })
            fig = px.bar(class_df, x='Class', y='Count', color='Class',
                        color_discrete_map={'Fake': '#ef5350', 'Real': '#66bb6a'})
            fig.update_layout(title='Article Count by Class', height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Text statistics
        if 'text_length' in df.columns:
            st.markdown("## 📝 Text Length Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Distribution by Class")
                fig = go.Figure()
                
                for label, name, color in [(0, 'Fake', '#ef5350'), (1, 'Real', '#66bb6a')]:
                    data = df[df['label'] == label]['text_length']
                    fig.add_trace(go.Box(y=data, name=name, marker_color=color))
                
                fig.update_layout(title='Text Length Distribution', height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### Statistics")
                stats_df = df.groupby('label')['text_length'].describe()
                stats_df.index = ['Fake', 'Real']
                st.dataframe(stats_df.style.format("{:.0f}"), use_container_width=True)
        
        # Word count analysis
        if 'word_count' in df.columns:
            st.markdown("## 📖 Word Count Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(df, x='word_count', color=df['label'].map({0: 'Fake', 1: 'Real'}),
                                  nbins=50, marginal='box',
                                  color_discrete_map={'Fake': '#ef5350', 'Real': '#66bb6a'})
                fig.update_layout(title='Word Count Distribution', height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                avg_words = df.groupby('label')['word_count'].mean()
                fig = go.Figure(data=[
                    go.Bar(x=['Fake', 'Real'], y=avg_words.values,
                          marker_color=['#ef5350', '#66bb6a'])
                ])
                fig.update_layout(title='Average Word Count', height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment analysis
        if 'sentiment_polarity' in df.columns:
            st.markdown("## 😊 Sentiment Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Sample data for scatter plot
                sample_size = min(1000, len(df))
                df_sample = df.sample(n=sample_size, random_state=42)
                
                fig = px.scatter(df_sample, 
                               x='sentiment_polarity', 
                               y='sentiment_subjectivity',
                               color=df_sample['label'].map({0: 'Fake', 1: 'Real'}),
                               color_discrete_map={'Fake': '#ef5350', 'Real': '#66bb6a'},
                               opacity=0.6)
                fig.update_layout(title='Sentiment Distribution (Polarity vs Subjectivity)', height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                sentiment_stats = df.groupby('label')[['sentiment_polarity', 'sentiment_subjectivity']].mean()
                sentiment_stats.index = ['Fake', 'Real']
                
                fig = go.Figure()
                fig.add_trace(go.Bar(name='Polarity', x=sentiment_stats.index, 
                                    y=sentiment_stats['sentiment_polarity']))
                fig.add_trace(go.Bar(name='Subjectivity', x=sentiment_stats.index,
                                    y=sentiment_stats['sentiment_subjectivity']))
                fig.update_layout(title='Average Sentiment Scores', barmode='group', height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Punctuation analysis
        if 'exclamation_count' in df.columns and 'question_count' in df.columns:
            st.markdown("## ❗ Punctuation Usage")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_exclaim = df.groupby('label')['exclamation_count'].mean()
                st.metric("Avg Exclamations (Fake)", f"{avg_exclaim[0]:.2f}")
                st.metric("Avg Exclamations (Real)", f"{avg_exclaim[1]:.2f}")
            
            with col2:
                avg_question = df.groupby('label')['question_count'].mean()
                st.metric("Avg Questions (Fake)", f"{avg_question[0]:.2f}")
                st.metric("Avg Questions (Real)", f"{avg_question[1]:.2f}")
            
            with col3:
                avg_caps = df.groupby('label')['capital_ratio'].mean()
                st.metric("Avg Capital Ratio (Fake)", f"{avg_caps[0]:.3f}")
                st.metric("Avg Capital Ratio (Real)", f"{avg_caps[1]:.3f}")
        
        # Key insights
        st.markdown("## 💡 Key Insights")
        
        with st.expander("🔍 View Insights"):
            st.markdown("""
            ### Characteristics of Fake News:
            - Often uses more exclamation marks and capital letters
            - May have more extreme sentiment (very positive or very negative)
            - Sometimes shorter or longer than typical news articles
            - Higher subjectivity in language
            - More sensationalist language patterns
            
            ### Characteristics of Real News:
            - More balanced and neutral tone
            - Consistent article length
            - Professional writing style
            - Lower use of attention-grabbing punctuation
            - More objective language
            
            ### Dataset Balance:
            - The dataset is well-balanced between fake and real news
            - This ensures the model doesn't have class bias
            - Both classes have sufficient examples for training
            """)
        
    except Exception as e:
        st.error(f"Error loading data: {e}")

else:
    st.warning("📁 Processed data not found!")
    st.info("Run preprocessing first: `python src/data_preprocessing.py`")
    
    # Show sample statistics
    st.markdown("## 📊 Expected Dataset Statistics")
    st.info("""
    Once you preprocess the data, you'll see:
    - Total article count (~45,000)
    - Class distribution (Fake vs Real)
    - Text length statistics
    - Word count distribution
    - Sentiment analysis
    - Punctuation patterns
    - And more insights!
    """)