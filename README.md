# 🔍 Fake News Detection System

> An AI-powered application that detects fake news articles using Natural Language Processing and Machine Learning with 95% accuracy.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](your-app-url-here)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Live Demo](#live-demo)
- [Screenshots](#screenshots)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a sophisticated fake news detection system that analyzes news articles and determines their credibility using advanced machine learning algorithms. The system combines **TF-IDF vectorization**, **statistical feature extraction**, and **sentiment analysis** to achieve high accuracy in distinguishing between fake and real news.

### Key Highlights

- ✅ **95% Accuracy** on test dataset
- 🚀 **Real-time predictions** with confidence scores
- 📊 **Batch processing** for multiple articles
- 🎨 **Interactive web interface** built with Streamlit
- 📈 **Model comparison** dashboard
- 🔒 **Privacy-focused** - no data stored

## ✨ Features

### 1. Single Article Analysis
- Paste any news article and get instant predictions
- Confidence scores with probability distribution
- Visual indicators for fake/real classification
- Detailed interpretation guidelines

### 2. Batch Prediction
- Upload CSV files with multiple articles
- Process hundreds of articles at once
- Download results with predictions
- Statistical summaries and visualizations

### 3. Model Insights
- Compare 5 different ML algorithms
- Performance metrics visualization
- ROC curves and confusion matrices
- Feature importance analysis

### 4. Smart Analysis
- **3,010 features** extracted per article
- **TF-IDF** text vectorization (unigrams + bigrams)
- **Statistical features**: length, punctuation, capitalization
- **Sentiment analysis**: polarity and subjectivity scores
- **Uncertainty handling** for ambiguous cases

## 🌐 Live Demo

🔗 **[Try the App Here](your-streamlit-app-url)**

*Replace with your actual Streamlit deployment URL*

## 📸 Screenshots

### Main Prediction Interface
![Main Interface](screenshots/main_interface.png)
*Clean, intuitive interface for analyzing news articles*

### Results Dashboard
![Results](screenshots/results.png)
*Detailed confidence analysis with visual indicators*

### Batch Processing
![Batch](screenshots/batch_processing.png)
*Process multiple articles from CSV files*

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data**
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"
```

5. **Prepare the dataset**

Place your dataset files in the `data/raw/` folder:
```
data/raw/
├── Fake.csv
└── True.csv
```

6. **Run preprocessing and training**
```bash
# Preprocess the data
python src/data_preprocessing.py

# Train the model
python src/model_training.py
```

7. **Launch the application**
```bash
streamlit run app/streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## 💻 Usage

### Single Article Prediction

1. Open the app in your browser
2. Paste a news article in the text area (minimum 100 words recommended)
3. Click **"Analyze Article"**
4. View results with confidence scores

### Batch Processing

1. Navigate to **"Batch Prediction"** page
2. Upload a CSV file with an article column
3. Select the column containing article text
4. Click **"Predict All Articles"**
5. Download results as CSV

### Sample CSV Format
```csv
id,article,source
1,"Breaking news text here...",NewsSource1
2,"Another article text...",Reuters
```

## 📊 Model Performance

### Overall Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 95.20% |
| **Precision (Fake)** | 94.85% |
| **Precision (Real)** | 95.10% |
| **Recall (Fake)** | 95.10% |
| **Recall (Real)** | 94.85% |
| **F1-Score** | 95.05% |
| **ROC-AUC** | 98.50% |

### Model Comparison

| Model | Accuracy | Training Time |
|-------|----------|---------------|
| **Logistic Regression** ⭐ | 95.20% | 45s |
| Random Forest | 92.30% | 480s |
| XGBoost | 91.80% | 320s |
| SVM | 89.70% | 180s |
| Naive Bayes | 86.20% | 15s |

⭐ *Current production model*

### Training Dataset
- **Total Articles**: 44,898
- **Fake News**: 23,481
- **Real News**: 21,417
- **Features**: 3,010 per article

## 📁 Project Structure

```
fake-news-detection/
│
├── data/
│   ├── raw/                    # Raw dataset files
│   │   ├── Fake.csv
│   │   └── True.csv
│   └── processed/              # Processed data
│       ├── cleaned_data.csv
│       └── data_stats.txt
│
├── models/                     # Trained models
│   ├── fake_news_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── scaler.pkl
│   └── model_metrics.txt
│
├── src/                        # Source code
│   ├── data_preprocessing.py   # Data cleaning & preprocessing
│   ├── feature_engineering.py  # Feature extraction
│   ├── model_training.py       # Model training
│   ├── model_evaluation.py     # Evaluation & visualization
│   └── utils.py                # Utility functions
│
├── app/                        # Streamlit application
│   ├── streamlit_app.py        # Main app
│   ├── pages/
│   │   ├── batch_prediction.py
│   │   └── model_comparison.py
│   └── assets/
│       └── style.css
│
├── .streamlit/                 # Streamlit config
│   └── config.toml
│
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── packages.txt                # System dependencies
├── .gitignore
└── README.md
```

## 🛠️ Technologies Used

### Machine Learning & NLP
- **scikit-learn** - ML algorithms and preprocessing
- **NLTK** - Natural language processing
- **TextBlob** - Sentiment analysis
- **XGBoost** - Gradient boosting
- **TF-IDF Vectorization** - Text feature extraction

### Data Processing
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **joblib** - Model serialization

### Visualization
- **Plotly** - Interactive charts
- **Matplotlib** - Static plots
- **Seaborn** - Statistical visualizations
- **WordCloud** - Text visualization

### Web Framework
- **Streamlit** - Web application framework

### Deployment
- **Docker** - Containerization
- **Streamlit Cloud** - Hosting

## 📚 Dataset

This project uses the **Fake and Real News Dataset** from Kaggle:

- **Source**: [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- **Size**: ~44,000 articles
- **Format**: CSV files (Fake.csv and True.csv)
- **License**: CC0: Public Domain

### Dataset Features
- `title` - Article headline
- `text` - Full article text
- `subject` - Article category
- `date` - Publication date

## 🐳 Docker Deployment

### Build Docker Image
```bash
docker build -t fake-news-detector .
```

### Run Container
```bash
docker run -p 8080:8080 fake-news-detector
```

Access at `http://localhost:8080`

## ☁️ Streamlit Cloud Deployment

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Set main file path: `app/streamlit_app.py`
5. Deploy!

**Important**: Make sure `Fake.csv` and `True.csv` are in your repository under `data/raw/` and the trained model files are in `models/`

## 🔧 Configuration

### Streamlit Config (`.streamlit/config.toml`)
```toml
[theme]
primaryColor="#667eea"
backgroundColor="#ffffff"
secondaryBackgroundColor="#f0f2f6"
textColor="#262730"

[server]
port = 8501
enableCORS = false
```

### Model Parameters
```python
# TF-IDF Configuration
max_features = 3000
ngram_range = (1, 2)
min_df = 5
max_df = 0.7

# Logistic Regression
C = 0.1  # Regularization
solver = 'saga'
max_iter = 1000
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Improvement
- [ ] Add more ML models (BERT, transformers)
- [ ] Implement REST API
- [ ] Add multilingual support
- [ ] Create mobile app
- [ ] Add real-time news fetching
- [ ] Improve real news detection accuracy

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Your Name** - *Initial work* - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- Dataset provided by [Clément Bisaillon](https://www.kaggle.com/clmentbisaillon) on Kaggle
- Streamlit team for the amazing framework
- scikit-learn contributors
- NLTK developers

## 📧 Contact

- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- **GitHub**: [@yourusername](https://github.com/yourusername)
- **Project Link**: [https://github.com/yourusername/fake-news-detection](https://github.com/yourusername/fake-news-detection)

## ⚠️ Disclaimer

This tool is for educational and research purposes. While it achieves high accuracy, it should not be the sole source for determining news credibility. Always cross-verify information with multiple reliable sources.

---

**Made with ❤️ and Python**

**Star ⭐ this repository if you found it helpful!**
