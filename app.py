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
                    results = classifier(month_data['Content'].tolist())
                    
                    # Process results
                    month_data['Sentiment'] = ["POSITIVE" if r['score'] > 0 else "NEGATIVE" for r in results]
                    # Get absolute probability score (Confidence)
                    month_data['Confidence'] = [round(abs(r['score']), 3) for r in results]
                    
                    # Create summary for the Bar Chart
                    summary = month_data.groupby('Sentiment').agg(
                        Review_Count=('Sentiment', 'count'),
                        Avg_Confidence=('Confidence', 'mean')
                    ).reset_index()

                    # --- Visualizations ---
                    st.subheader(f"Analysis Results for {months[selected_month_num-1]} 2023")
                    
                    # Requirement: Bar Chart showing Count and Confidence Score
                    fig = px.bar(
                        summary, 
                        x='Sentiment', 
                        y='Review_Count',
                        color='Sentiment',
                        text='Review_Count',
                        hover_data=['Avg_Confidence'], # Requirement: Tooltip indicates Avg Confidence
                        labels={'Review_Count': 'Number of Reviews', 'Avg_Confidence': 'Avg. AI Confidence'},
                        color_discrete_map={'POSITIVE':'#2ecc71', 'NEGATIVE':'#e74c3c'}
                    )
                    
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)

                    # Display average confidence in a metric for extra points
                    avg_conf = month_data['Confidence'].mean()
                    st.metric("Overall Model Confidence", f"{avg_conf:.2%}")

                    st.table(month_data[['Content', 'Sentiment', 'Confidence']])
                        
                except Exception as e:
                    st.error("Memory limit reached. Try refreshing or picking a month with fewer reviews.")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.caption("📊 **Data Mining Assignment**")
st.sidebar.caption("Deployed on Render (Free Tier)")