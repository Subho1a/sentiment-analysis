# 🐦 Twitter Sentiment Analysis Using Machine Learning

A comprehensive machine learning project that analyzes Twitter sentiment using scikit-learn, with a Streamlit web application for real-time predictions.

## 📋 Project Overview

This project implements a multi-model sentiment analysis system that classifies tweets into three categories:
- **Negative 😞** - Negative sentiment
- **Neutral 😐** - Neutral sentiment  
- **Positive 😊** - Positive sentiment

The project uses advanced NLP techniques, hyperparameter tuning, and ensemble methods to achieve high accuracy predictions.

## ✨ Features

- **Text Preprocessing Pipeline**
  - URL and HTML tag removal
  - User mention and hashtag processing
  - Lemmatization and stopword removal
  - Custom text cleaning functions

- **Multiple Model Training**
  - Logistic Regression
  - Support Vector Machines (SVM)
  - Random Forest
  - Naive Bayes
  - Ensemble Voting Classifier

- **Advanced Optimization**
  - TF-IDF vectorization with N-grams
  - Hyperparameter tuning with GridSearchCV
  - Class imbalance handling with balanced weights
  - Cross-validation analysis
  - Ensemble voting methods

- **Interactive Web Application**
  - Real-time sentiment prediction
  - Confidence score visualization
  - Text preprocessing visualization
  - Dataset analytics and insights
  - Beautiful UI with color-coded results

## 📊 Dataset

The project uses Twitter sentiment datasets with multiple CSV files:

```
DataSet/
├── train.csv                                    # Main training data
├── test.csv                                     # Test data for evaluation
├── testdata.manual.2009.06.14.csv              # Manual test data
└── training.1600000.processed.noemoticon.csv   # Large training dataset (1.6M tweets)
```

**Dataset Format:**
- `text`: Tweet content
- `sentiment`: Sentiment label (0=Negative, 2=Neutral, 4=Positive)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Subho1a/sentiment-analysis.git
cd sentiment-analysis
```

2. **Create and activate virtual environment**
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Project

#### Option 1: Jupyter Notebook (Training & Analysis)
```bash
jupyter notebook twitter-sentiment-analysis-with-sklearn.ipynb
```

The notebook includes:
- Data loading and preprocessing
- Exploratory data analysis (EDA)
- Model training and comparison
- Hyperparameter optimization
- Cross-validation analysis
- Test set evaluation

#### Option 2: Streamlit Web App (Real-time Predictions)
```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

## 📁 Project Structure

```
Sentiment-Analysis-Using-ML/
├── README.md                                    # This file
├── requirements.txt                             # Project dependencies
├── app.py                                       # Streamlit web application
├── twitter-sentiment-analysis-with-sklearn.ipynb # Main Jupyter notebook
├── .gitignore                                   # Git ignore rules
├── DataSet/                                     # Dataset directory
│   ├── train.csv
│   ├── test.csv
│   ├── testdata.manual.2009.06.14.csv
│   └── training.1600000.processed.noemoticon.csv
└── models/                                      # Trained models (generated after training)
    ├── sentiment_model.pkl                      # Best trained model
    ├── label_encoder.pkl                        # Label encoder
    └── metadata.pkl                             # Model metadata
```

## 🤖 Model Performance

### Original Models
- Logistic Regression (Count Vectors): ~78%
- Logistic Regression (TF-IDF): ~79%
- Naive Bayes: ~77%
- Linear SVC: ~80%
- Random Forest: ~75%

### Improved Models
- **Logistic Regression + NGrams + Grid Search**: ~69-70%
- **Random Forest + NGrams**: ~68-69%
- **Ensemble Voting Classifier**: ~69.14% ⭐

### Optimization Techniques Applied
✅ N-grams (bigrams for phrase capture)  
✅ Increased vocabulary (5,000 → 10,000 features)  
✅ Class balancing (for imbalanced data)  
✅ GridSearchCV hyperparameter tuning  
✅ 5-fold cross-validation  
✅ Ensemble voting methods  

## 📊 Text Preprocessing Pipeline

1. **Lowercasing** - Convert to lowercase
2. **URL Removal** - Remove URLs and web links
3. **HTML Cleaning** - Remove HTML tags
4. **Mention Removal** - Remove @mentions
5. **Hashtag Processing** - Extract text from hashtags
6. **Punctuation Removal** - Remove special characters
7. **Number Removal** - Remove digits
8. **Tokenization** - Split into words
9. **Stopword Removal** - Remove common words
10. **Lemmatization** - Reduce words to base form

## 🎯 Using the Streamlit App

### Home Page
- Model accuracy and dataset statistics
- Sentiment distribution visualization
- Sample tweets preview

### Predict Page
- Input custom text or select sample tweets
- Real-time sentiment prediction
- Confidence scores for each class
- Text preprocessing visualization

### Analytics Page
- Sentiment distribution charts
- Tweet length analysis
- Dataset statistics
- Most common words by sentiment

## 💾 Model Persistence

Models are saved as pickle files for quick loading:

```python
# Load the model in app.py
with open('models/sentiment_model.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Make predictions
prediction = pipeline.predict([processed_text])
probabilities = pipeline.predict_proba([processed_text])
```

## 🔧 Technologies Used

**Core Libraries:**
- `scikit-learn` - Machine learning models
- `pandas` - Data manipulation
- `numpy` - Numerical computing

**NLP Tools:**
- `nltk` - Natural language toolkit
- `regex` - Advanced text processing

**Web Framework:**
- `streamlit` - Interactive web app
- `matplotlib` - Visualization
- `seaborn` - Statistical plots

## 📈 Notebook Cells Overview

| # | Cell | Purpose |
|---|------|---------|
| 1-16 | Data Loading & EDA | Load data and exploratory analysis |
| 17-23 | Text Preprocessing | Define cleaning and preprocessing functions |
| 24-31 | Data Preparation | Encode labels and split data |
| 32-40 | Original Models | Train baseline models |
| 41-45 | Class Imbalance Analysis | Check data distribution |
| 46-48 | Improved Models | Train enhanced models with ngrams |
| 49 | GridSearchCV | Hyperparameter optimization |
| 50 | Model Comparison | Compare all models |
| 51-56 | Advanced Methods | Ensemble voting and cross-validation |
| 57 | Test Evaluation | Evaluate on test set |
| 58-59 | Final Comparison & Save | Select best model and save |

## 🚀 Future Improvements

- [ ] Add Word2Vec/GloVe embeddings
- [ ] Implement deep learning models (LSTM, CNN)
- [ ] Use pre-trained transformers (BERT, DistilBERT)
- [ ] Add Docker containerization
- [ ] Deploy to cloud (AWS, GCP, Heroku)
- [ ] Add API endpoints for model serving
- [ ] Implement batch prediction
- [ ] Add model explainability (LIME, SHAP)
- [ ] Support for other languages
- [ ] Real-time Twitter API integration

## 📝 Usage Examples

### Python Script
```python
from sklearn.pipeline import Pipeline
import pickle

# Load model
with open('models/sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make prediction
tweet = "I love this amazing product! Highly recommend!"
prediction = model.predict([tweet])
probabilities = model.predict_proba([tweet])

print(f"Sentiment: {prediction[0]}")  # 2 = Positive
print(f"Confidence: {probabilities[0]}")
```

### Streamlit App
```bash
# Start the app
streamlit run app.py

# Open browser to http://localhost:8501
```

## 📊 Results Summary

| Metric | Value |
|--------|-------|
| Final Model Accuracy | 69.14% |
| Test Set Accuracy | 69.75% |
| Best Model | Ensemble Voting or GridSearch Optimized LR |
| Training Time | ~5-10 minutes |
| Prediction Time (per tweet) | <10ms |

## 🐛 Troubleshooting

**Issue: ModuleNotFoundError**
```bash
pip install -r requirements.txt
```

**Issue: NLTK data missing**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

**Issue: Models directory not found**
- Run the notebook completely to generate pickle files
- Or manually create `models/` directory

**Issue: CPU high usage during training**
- Reduce `n_jobs` parameter in GridSearchCV
- Use smaller dataset for testing

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [NLTK Documentation](https://www.nltk.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Twitter Sentiment Analysis Papers](https://scholar.google.com/)

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

**Arindam Jana**
- GitHub: [@Subho1a](https://github.com/Subho1a)
- Email: arindamjana693@gmail.com

## 🙏 Acknowledgments

- Twitter Sentiment Dataset providers
- Scikit-learn and open-source ML community
- Streamlit team for the amazing framework

---

**Built with ❤️ using Machine Learning and NLP techniques**

Last Updated: April 18, 2026
