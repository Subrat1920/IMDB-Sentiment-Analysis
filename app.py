import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
import random

# Load IMDB word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load model
model = load_model("Notebook\keras_models\sentiment_model.keras")

# Movie stills URLs (replace with your own images)
MOVIE_STILLS = [
    "https://images.unsplash.com/photo-1536440136628-849c177e76a1?ixlib=rb-1.2.1&auto=format&fit=crop&w=1920&q=80",  # Cinema
    "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?ixlib=rb-1.2.1&auto=format&fit=crop&w=1920&q=80",  # Movie theater
    "https://images.pexels.com/photos/7991579/pexels-photo-7991579.jpeg?auto=compress&cs=tinysrgb&w=1920&h=1080&dpr=2",  # Film reel
    "https://images.pexels.com/photos/7991158/pexels-photo-7991158.jpeg?auto=compress&cs=tinysrgb&w=1920&h=1080&dpr=2",  # Popcorn
    "https://images.unsplash.com/photo-1517604931442-7e0c8ed2963c?ixlib=rb-1.2.1&auto=format&fit=crop&w=1920&q=80",  # Director's chair
]
# Function to preprocess text
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = pad_sequences([encoded_review], maxlen=500, padding='pre')
    return padded_review

# Function to predict sentiment
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)[0][0]
    confidence = abs(prediction - 0.5) * 200
    return prediction, confidence

# Streamlit UI Configuration
st.set_page_config(
    page_title="CineMood AI",
    layout="wide",
    page_icon="🎬",
    initial_sidebar_state="collapsed"
)

# Custom CSS with cinematic theme
st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&family=Cinzel:wght@700&display=swap');
        
        .main {{
            background: #0a0a0a;
            color: #ffffff;
            font-family: 'Montserrat', sans-serif;
        }}
        
        .header {{
            background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)),
                        url({random.choice(MOVIE_STILLS)});
            background-size: cover;
            background-position: center;
            padding: 8rem 2rem;
            text-align: center;
            margin: -2rem -2rem 2rem -2rem;
        }}
        
        .stTextArea textarea {{
            background: rgba(255,255,255,0.1);
            color: white!important;
            border: 2px solid #ffd700;
            border-radius: 10px;
            padding: 1.5rem;
            font-size: 1.1rem;
            transition: all 0.3s ease;
        }}
        
        .stButton>button {{
            background: linear-gradient(45deg, #ffd700, #ff6b00);
            color: #000000!important;
            border: none;
            border-radius: 30px;
            padding: 1rem 2.5rem;
            font-size: 1.2rem;
            font-weight: bold;
            transition: all 0.3s ease;
            font-family: 'Cinzel', serif;
        }}
        
        .movie-stills {{
            display: flex;
            overflow-x: auto;
            gap: 1rem;
            padding: 2rem 0;
            margin: 2rem 0;
        }}
        
        .movie-still {{
            min-width: 300px;
            height: 200px;
            border-radius: 10px;
            transition: transform 0.3s ease;
        }}
        
        .movie-still:hover {{
            transform: scale(1.05);
        }}
        
        .result-card {{
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 2rem;
            margin: 2rem 0;
            border: 2px solid;
            position: relative;
            overflow: hidden;
        }}
        
        .result-card::before {{
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(45deg, transparent, rgba(255,215,0,0.1), transparent);
            transform: rotate(45deg);
            animation: shine 3s infinite;
        }}
        
        @keyframes shine {{
            0% {{ transform: rotate(45deg) translateX(-150%); }}
            100% {{ transform: rotate(45deg) translateX(150%); }}
        }}
        
        .confidence-meter {{
            height: 15px;
            background: rgba(255,255,255,0.2);
            border-radius: 10px;
            overflow: hidden;
            position: relative;
        }}
        
        .confidence-fill {{
            height: 100%;
            transition: width 0.5s ease;
            background: linear-gradient(90deg, #ff0000, #ffd700, #00ff00);
        }}
    </style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown(f"""
    <div class="header">
        <h1 style="font-family: 'Cinzel', serif; font-size: 3.5rem; color: #ffd700;">
            CINEMOOD AI
        </h1>
        <p style="font-size: 1.5rem; color: #ffffff;">
            Where Artificial Intelligence Meets Cinematic Passion
        </p>
    </div>
""", unsafe_allow_html=True)

# Movie Stills Carousel
st.markdown("""
    <div class="movie-stills">
        <img class="movie-still" src="https://images.unsplash.com/photo-1536440136628-849c177e76a1?ixlib=rb-1.2.1&auto=format&fit=crop&w=1920&q=80">
        <img class="movie-still" src="https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?ixlib=rb-1.2.1&auto=format&fit=crop&w=1920&q=80">
        <img class="movie-still" src="https://images.pexels.com/photos/7991579/pexels-photo-7991579.jpeg?auto=compress&cs=tinysrgb&w=1920&h=1080&dpr=2">
        <img class="movie-still" src="https://images.pexels.com/photos/7991158/pexels-photo-7991158.jpeg?auto=compress&cs=tinysrgb&w=1920&h=1080&dpr=2">
        <img class="movie-still" src="https://images.unsplash.com/photo-1517604931442-7e0c8ed2963c?ixlib=rb-1.2.1&auto=format&fit=crop&w=1920&q=80">
    </div>
""", unsafe_allow_html=True)

# Main Content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
        <div style="padding: 2rem; background: rgba(255,215,0,0.1); border-radius: 15px;">
            <h2 style="color: #ffd700; font-family: 'Cinzel';">🎭 The Art of Cinematic Analysis</h2>
            <p style="font-size: 1.1rem;">
                Our AI has been trained on thousands of movie reviews to understand the subtle nuances 
                of cinematic critique. From blockbuster hits to indie gems, CineMood AI deciphers the 
                emotional core of your review with Hollywood-level precision.
            </p>
            <img src="https://images.pexels.com/photos/8386440/pexels-photo-8386440.jpeg?auto=compress&cs=tinysrgb&w=1920&h=1080&dpr=2" 
                 style="width: 100%; border-radius: 10px; margin-top: 1rem;">
        </div>
    """, unsafe_allow_html=True)

with col2:
    review = st.text_area(
        "✨ WRITE YOUR MOVIE REVIEW:",
        height=200,
        placeholder="This film transported me to another world...",
        key="review_input"
    )

    if st.button("🎥 ANALYZE CINEMATIC IMPACT", use_container_width=True):
        if not review.strip():
            st.error("Please enter a review to analyze!")
        else:
            with st.spinner("Reel is spinning... capturing cinematic essence..."):
                prediction, confidence = predict_sentiment(review)
                sentiment_color = "#00ff00" if prediction >= 0.5 else "#ff0000"
                sentiment_emoji = "🎬" if prediction >= 0.5 else "💥"
                sentiment_label = "CINEMATIC MASTERPIECE" if prediction >= 0.5 else "BOX OFFICE BOMB"

                st.markdown(f"""
                    <div class="result-card" style="border-color: {sentiment_color}">
                        <div style="text-align: center;">
                            <div style="font-size: 4rem; margin-bottom: 1rem;">
                                {sentiment_emoji}
                            </div>
                            <h2 style="color: {sentiment_color}; margin: 1rem 0;">
                                {sentiment_label}
                            </h2>
                            <div class="confidence-meter">
                                <div class="confidence-fill" style="width: {min(confidence, 100)}%;"></div>
                            </div>
                            <p style="font-size: 1.2rem; margin: 1rem 0;">
                                AI Confidence: {min(confidence, 100):.1f}%
                            </p>
                            <div style="display: flex; justify-content: space-between; color: #ffffff;">
                                <span>😡 Negative</span>
                                <span>🎭 Neutral</span>
                                <span>😍 Positive</span>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

# Example Section
st.markdown("""
    <div style="background: rgba(255,215,0,0.1); padding: 2rem; border-radius: 15px; margin-top: 2rem;">
        <h3 style="color: #ffd700; font-family: 'Cinzel';">🍿 Try These Classic Reviews</h3>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1rem;">
            <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px;">
                <div style="color: #00ff00;">🌟 "A masterpiece of modern cinema!"</div>
                <div style="font-size: 0.9rem; color: #888;">(The Godfather, 1972)</div>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px;">
                <div style="color: #ff0000;">💥 "A complete disaster from start to finish"</div>
                <div style="font-size: 0.9rem; color: #888;">(Battlefield Earth, 2000)</div>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px;">
                <div style="color: #ffd700;">🎭 "Beautiful visuals but lacks depth"</div>
                <div style="font-size: 0.9rem; color: #888;">(Avatar, 2009)</div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)