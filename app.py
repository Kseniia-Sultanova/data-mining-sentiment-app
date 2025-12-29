import streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.express as px

# --- 1. Page Config ---
st.set_page_config(page_title="Brand Reputation 2023", layout="wide")

# --- 2. Cache AI Model (Saves memory) ---
@st.cache_resource
def load_sentiment_model():
    # Using a small, efficient model to stay within Render's free RAM limits
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# --- 3. Load Scraped Data ---
@st.cache_data
def load_data():
    # Ensure this CSV is in the same folder as app.py
    df = pd.read_csv("scraped_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Initialize Data and Model
try:
    df = load_data()
    sentiment_analyzer = load_sentiment_model()
except Exception as e:
    st.error(f"Error loading data or model: {e}")
    st.stop()

# --- 4. Sidebar Navigation ---
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to:", ["📦 Product Catalog", "⭐ 2023 Sentiment Analysis"])

# --- 5. Page: Product Catalog ---
if menu == "📦 Product Catalog":
    st.title("Product Catalog")
    st.write("List of products extracted from the sandbox environment.")
    
    product_df = df[df['Type'] == 'Product']
    st.dataframe(product_df[['Title', 'Content']], use_container_width=True)

# --- 6. Page: Sentiment Analysis ---
else:
    st.title("⭐ 2023 Review Sentiment Analysis")
    st.write("Use the slider to analyze customer sentiment for specific months in 2023.")

    # Month Slider (Mandatory Requirement)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    selected_month_num = st.select_slider(
        "Select Month (2023)", 
        options=range(1, 13), 
        format_func=lambda x: months[x-1]
    )

    # Filtering data for the selected month
    mask = (df['Type'] == 'Review') & (df['Date'].dt.month == selected_month_num)
    filtered_df = df[mask].copy()

    if filtered_df.empty:
        st.warning(f"No reviews available for {months[selected_month_num-1]} 2023.")
    else:
        # Run AI Sentiment Analysis
        with st.spinner("Analyzing sentiments..."):
            results = sentiment_analyzer(filtered_df['Content'].tolist())
            filtered_df['Sentiment'] = [res['label'] for res in results]
            filtered_df['Score'] = [round(res['score'], 3) for res in results]

        # Visualization
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Sentiment Distribution")
            fig = px.pie(
                filtered_df, 
                names='Sentiment', 
                color='Sentiment',
                color_discrete_map={'POSITIVE':'#00CC96', 'NEGATIVE':'#EF553B'}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Data Details")
            st.dataframe(filtered_df[['Content', 'Sentiment', 'Score']], use_container_width=True)

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.info("Project for Economics & Data Science. Scraped from web-scraping.dev.")