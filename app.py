import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import warnings
from datetime import datetime
import regex as re
import string
warnings.filterwarnings('ignore')

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

download_nltk_resources()

# Text preprocessing functions
def clean_text(text):
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove user mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags (keeping the text after #)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_text(text):
    if not text:
        return ""
    
    # Clean text
    text = clean_text(text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back
    processed_text = ' '.join(tokens)
    
    return processed_text

# Load model from pickle files
@st.cache_resource
def load_model():
    import os
    
    model_path = "models/sentiment_model.pkl"
    encoder_path = "models/label_encoder.pkl"
    metadata_path = "models/metadata.pkl"
    
    # Check if model files exist
    if not all(os.path.exists(p) for p in [model_path, encoder_path, metadata_path]):
        st.error("❌ Model files not found! Please run the notebook first to generate the pickle files.")
        st.stop()
    
    # Load the trained model
    with open(model_path, 'rb') as f:
        pipeline = pickle.load(f)
    
    # Load the label encoder
    with open(encoder_path, 'rb') as f:
        le = pickle.load(f)
    
    # Load metadata
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    # Load dataset for analytics
    path = "DataSet"
    try:
        train_df = pd.read_csv(f'{path}/train.csv', encoding='latin1')
        train_df = train_df.dropna()
    except:
        train_df = pd.DataFrame()
    
    return pipeline, le, metadata['accuracy'], train_df, metadata

# Streamlit configuration
st.set_page_config(
    page_title="Twitter Sentiment Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.8rem;
        border: 2px solid #e0e0e0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .metric-label {
        font-weight: bold;
        font-size: 16px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Settings")
page = st.sidebar.radio("Navigate", ["🏠 Home", "🔮 Predict", "📈 Analytics"])

# Load model
pipeline, label_encoder, model_accuracy, train_data, metadata = load_model()

# Sentiment mapping - Updated for better visibility
sentiment_map = {0: 'Negative 😞', 1: 'Neutral 😐', 2: 'Positive 😊'}
sentiment_colors = {
    0: '#FF6B6B',  # Bright red for negative
    1: '#FFD93D',  # Bright yellow for neutral
    2: '#6BCB77'   # Bright green for positive
}

if page == "🏠 Home":
    st.title("🐦 Twitter Sentiment Analysis")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div style='
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                color: white;
            '>
                <div style='font-size: 14px; margin-bottom: 10px;'>Model Accuracy</div>
                <div style='font-size: 32px; font-weight: bold;'>{model_accuracy*100:.2f}%</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div style='
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                color: white;
            '>
                <div style='font-size: 14px; margin-bottom: 10px;'>Training Samples</div>
                <div style='font-size: 32px; font-weight: bold;'>{len(train_data):,}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div style='
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                color: white;
            '>
                <div style='font-size: 14px; margin-bottom: 10px;'>Dataset Classes</div>
                <div style='font-size: 32px; font-weight: bold;'>{len(train_data['sentiment'].unique())}</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("📊 Dataset Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Sentiment Distribution:**")
        sentiment_counts = train_data['sentiment'].value_counts()
        st.bar_chart(sentiment_counts)
    
    with col2:
        st.write("**Sample Data:**")
        st.dataframe(train_data[['text', 'sentiment']].head(10), use_container_width=True)
    
    st.info("✨ Use the **Predict** tab to analyze new tweets!")

elif page == "🔮 Predict":
    st.title("🔮 Sentiment Prediction")
    st.markdown("---")
    
    # Input options
    input_method = st.radio("Choose input method:", ["Type Text", "Sample Tweets"])
    
    if input_method == "Type Text":
        user_input = st.text_area("Enter tweet or text:", height=150, placeholder="Type or paste a tweet here...")
    else:
        sample_tweets = [
            "I love this product! It's amazing!",
            "This is just okay, nothing special.",
            "Worst experience ever. Very disappointed.",
            "Absolutely fantastic! Highly recommend!",
            "Not what I expected. Disappointed with the quality."
        ]
        selected_sample = st.selectbox("Select a sample tweet:", sample_tweets)
        user_input = selected_sample
    
    if st.button("Predict Sentiment", type="primary", use_container_width=True):
        if user_input.strip():
            # Preprocess and predict
            processed_text = preprocess_text(user_input)
            prediction = pipeline.predict([processed_text])[0]
            probabilities = pipeline.predict_proba([processed_text])[0]
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Prediction Result")
            
            sentiment_label = sentiment_map[prediction]
            sentiment_color = sentiment_colors[prediction]
            
            # Display sentiment with color
            background_color = sentiment_color
            st.markdown(f"""
                <div style='
                    background-color: {background_color}; 
                    padding: 30px; 
                    border-radius: 10px; 
                    text-align: center;
                    color: white;
                    font-size: 36px;
                    font-weight: bold;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                    margin-bottom: 30px;
                '>
                {sentiment_label}
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("Confidence Scores")
            
            # Display confidence scores with better styling and distinct colors
            col1, col2, col3 = st.columns(3)
            
            metrics_info = [
                ("😞 Negative", f"{probabilities[0]*100:.1f}%", "#E74C3C"),
                ("😐 Neutral", f"{probabilities[1]*100:.1f}%", "#3498DB"),
                ("😊 Positive", f"{probabilities[2]*100:.1f}%", "#27AE60")
            ]
            
            cols = [col1, col2, col3]
            for idx, (col, (label, value, color)) in enumerate(zip(cols, metrics_info)):
                with col:
                    st.markdown(f"""
                        <div style='
                            background-color: {color}; 
                            padding: 20px; 
                            border-radius: 10px;
                            text-align: center;
                            color: white;
                            font-weight: bold;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
                        '>
                            <div style='font-size: 16px; margin-bottom: 10px;'>{label}</div>
                            <div style='font-size: 28px;'>{value}</div>
                        </div>
                    """, unsafe_allow_html=True)
            
            
            # Show confidence bar
            st.markdown("---")
            st.subheader("📊 Confidence Distribution")
            confidence_data = pd.DataFrame({
                'Sentiment': ['Negative 😞', 'Neutral 😐', 'Positive 😊'],
                'Confidence': probabilities
            })
            st.bar_chart(confidence_data.set_index('Sentiment'))
            
            
            # Show processing steps
            st.markdown("---")
            with st.expander("📝 View Processing Steps"):
                st.write("**Original Text:**")
                st.write(user_input)
                st.write("\n**Cleaned Text:**")
                st.write(clean_text(user_input))
                st.write("\n**Processed Text:**")
                st.write(processed_text)
        else:
            st.warning("⚠️ Please enter some text to analyze!")

elif page == "📈 Analytics":
    st.title("📈 Analytics & Insights")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sentiment Distribution")
        sentiment_counts = train_data['sentiment'].value_counts()
        st.bar_chart(sentiment_counts)
    
    with col2:
        st.subheader("Tweet Length Distribution")
        train_data['text_length'] = train_data['text'].apply(len)
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.hist(train_data['text_length'], bins=50, color='skyblue', edgecolor='black')
        ax.set_xlabel('Tweet Length (characters)')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)
    
    st.subheader("Dataset Statistics")
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    
    stats_data = [
        ("Total Tweets", f"{len(train_data):,}", "#FF6B6B"),
        ("Avg Tweet Length", f"{train_data['text'].str.len().mean():.0f} chars", "#FFD93D"),
        ("Max Tweet Length", f"{train_data['text'].str.len().max():,} chars", "#6BCB77"),
        ("Min Tweet Length", f"{train_data['text'].str.len().min()} chars", "#4A90E2")
    ]
    
    for col, (label, value, color) in zip([stats_col1, stats_col2, stats_col3, stats_col4], stats_data):
        with col:
            st.markdown(f"""
                <div style='
                    background-color: {color};
                    padding: 15px;
                    border-radius: 8px;
                    text-align: center;
                    color: white;
                    font-weight: bold;
                '>
                    <div style='font-size: 12px; margin-bottom: 8px;'>{label}</div>
                    <div style='font-size: 20px;'>{value}</div>
                </div>
            """, unsafe_allow_html=True)
    
    # Top words
    st.subheader("Most Common Words by Sentiment")
    
    from collections import Counter
    
    sentiment_types = train_data['sentiment'].unique()
    selected_sentiment = st.selectbox("Select sentiment:", sentiment_types)
    
    # Get texts for selected sentiment and preprocess them
    sentiment_texts = train_data[train_data['sentiment'] == selected_sentiment]['text'].iloc[:5000]  # Limit to 5000 for performance
    
    # Preprocess texts to get processed versions
    processed_texts = sentiment_texts.apply(preprocess_text)
    all_words = ' '.join(processed_texts).split()
    word_freq = Counter(all_words).most_common(15)
    
    if word_freq:
        words, freqs = zip(*word_freq)
        word_df = pd.DataFrame({'Word': words, 'Frequency': freqs})
        st.bar_chart(word_df.set_index('Word'))
    else:
        st.info("No data available for this sentiment")
