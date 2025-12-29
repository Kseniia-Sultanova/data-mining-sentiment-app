import streamlit as st
import pandas as pd
import transformers
from transformers import pipeline
import plotly.express as px

# Force check if pipeline is available
if not hasattr(transformers, 'pipeline'):
    st.error("Library initialization error. Please refresh the page.")
    st.stop()

# --- 1. Page Configuration ---
st.set_page_config(page_title="Brand Reputation Dashboard", layout="wide")

# --- 2. Lightweight AI Model Loading ---
@st.cache_resource
def load_sentiment_model():
    # This model is ~80MB, perfect for Render's 512MB RAM limit
    return pipeline(
        "sentiment-analysis", 
        model="cross-encoder/ms-marco-MiniLM-L-2-v2"
    )

# --- 3. Data Loading ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("scraped_data.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

# Initialize app data
df = load_data()

# --- 4. Sidebar Navigation ---
st.sidebar.title("🔍 Project Menu")
page = st.sidebar.radio("Navigate to:", ["Product Catalog", "2023 Sentiment Analysis"])

# --- 5. Page: Product Catalog ---
if page == "Product Catalog":
    st.title("📦 Scraped Product Catalog")
    st.write("This data was extracted live from the sandbox environment.")
    
    product_df = df[df['Type'] == 'Product']
    if not product_df.empty:
        st.dataframe(product_df[['Title', 'Content']], use_container_width=True)
    else:
        st.warning("No product data found in CSV.")

# --- 6. Page: Sentiment Analysis ---
else:
    st.title("⭐ 2023 Customer Sentiment Analysis")
    st.write("Analyze monthly trends using AI (MiniLM Transformer).")

    # Month Slider (Mandatory Requirement)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    selected_month_num = st.select_slider(
        "Select Month (2023)", 
        options=range(1, 13), 
        format_func=lambda x: months[x-1]
    )

    # Filter data for selected month
    month_data = df[(df['Type'] == 'Review') & (df['Date'].dt.month == selected_month_num)].copy()

    if month_data.empty:
        st.info(f"No reviews recorded for {months[selected_month_num-1]} 2023.")
    else:
        st.write(f"Found **{len(month_data)}** reviews for this period.")
        
        # Action Button to save RAM
        if st.button("🚀 Run AI Analysis"):
            with st.spinner("AI is processing text..."):
                try:
                    classifier = load_sentiment_model()
                    # Run analysis
                    texts = month_data['Content'].tolist()
                    results = classifier(texts)
                    
                    # MiniLM scores: higher score usually indicates better match/positivity
                    # We map the score to a binary label for the chart
                    month_data['Sentiment'] = ["POSITIVE" if r['score'] > 0 else "NEGATIVE" for r in results]
                    
                    # --- Visualizations ---
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader("Review Volume by Sentiment")
                        # Creating a summary dataframe for the bar chart
                        sentiment_counts = month_data['Sentiment'].value_counts().reset_index()
                        sentiment_counts.columns = ['Sentiment', 'Count']
                        
                        fig = px.bar(
                            sentiment_counts, 
                            x='Sentiment', 
                            y='Count',
                            color='Sentiment',
                            text='Count',
                            color_discrete_map={'POSITIVE':'#2ecc71', 'NEGATIVE':'#e74c3c'}
                        )
                        fig.update_traces(textposition='outside')
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        st.subheader("Extracted Insights")
                        st.table(month_data[['Content', 'Sentiment']])
                        
                except Exception as e:
                    st.error("The AI model ran out of memory. Try a different month or refresh.")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.caption("📊 **Data Mining Assignment**")
st.sidebar.caption("Deployed on Render (Free Tier)")