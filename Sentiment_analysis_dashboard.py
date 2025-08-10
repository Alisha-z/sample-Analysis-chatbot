import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
from transformers import pipeline

# ----------- Sidebar Summary ----------- #
st.sidebar.markdown("## Dashboard Info")
st.sidebar.info(
    """
    **Model:** DistilBERT (Hugging Face)

    **Features:**
    - Upload your own CSV file
    - Filter reviews by sentiment
    - Keyword search
    - Download filtered results
    - Interactive charts & wordclouds
    - Real-time sentiment prediction with confidence score
    """
)


# ----------- Load Hugging Face Model (PyTorch backend) ----------- #
@st.cache_resource
def get_hf_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        framework="pt"
    )


hf_model = get_hf_sentiment_model()


def get_sentiment_hf(text):
    result = hf_model(text)[0]
    label = result['label']
    score = result['score']
    sentiment = "Positive" if label == "POSITIVE" and score > 0.6 else "Negative" if label == "NEGATIVE" and score > 0.6 else "Neutral"
    return sentiment, score


st.title("Advanced Sentiment Analysis Dashboard")

uploaded_file = st.sidebar.file_uploader("Upload CSV (column: 'review')", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.DataFrame({
        "review": [
            "I love this movie, it was fantastic!",
            "The product broke after one use.",
            "Service was okay, nothing special.",
            "Absolutely wonderful experience.",
            "Not worth the money.",
            "Will buy again, great quality!",
            "Terrible customer support.",
            "Decent, but could be better.",
            "Exceeded my expectations!",
            "Horrible taste, will not recommend."
        ]
    })

if 'Sentiment' not in df.columns or 'Confidence' not in df.columns:
    with st.spinner("Analyzing sentiments..."):
        df[['Sentiment', 'Confidence']] = df['review'].astype(str).apply(lambda x: pd.Series(get_sentiment_hf(x)))

sentiment_options = ["All"] + sorted(df['Sentiment'].unique())
selected_sentiment = st.sidebar.selectbox("Filter by Sentiment", sentiment_options)
search_term = st.sidebar.text_input("Keyword Search")

filtered_df = df.copy()
if selected_sentiment != "All":
    filtered_df = filtered_df[filtered_df['Sentiment'] == selected_sentiment]
if search_term:
    filtered_df = filtered_df[filtered_df['review'].str.contains(search_term, case=False)]

st.subheader("Filtered Reviews")
st.dataframe(filtered_df[['review', 'Sentiment', 'Confidence']], use_container_width=True)

st.sidebar.download_button(
    "Download Filtered Data",
    filtered_df.to_csv(index=False),
    "filtered_reviews.csv"
)

stats = df['Sentiment'].value_counts().reset_index()
stats.columns = ["Sentiment", "Count"]

st.subheader("Sentiment Distribution")
fig_pie = px.pie(stats, names='Sentiment', values='Count', title='Sentiment Pie Chart', color='Sentiment')
st.plotly_chart(fig_pie, use_container_width=True)

fig_bar = px.bar(stats, x='Sentiment', y='Count', color='Sentiment', title='Sentiment Bar Chart')
st.plotly_chart(fig_bar, use_container_width=True)

st.subheader("Word Clouds by Sentiment")
cols = st.columns(len(stats))
for idx, sentiment in enumerate(stats['Sentiment']):
    text = " ".join(df[df['Sentiment'] == sentiment]['review'])
    wc = WordCloud(width=400, height=200, background_color='white').generate(text) if text else None
    with cols[idx]:
        st.markdown(f"**{sentiment}**")
        if wc:
            st.image(wc.to_array())
        else:
            st.write("No reviews.")

st.subheader("Sample Reviews")
sample_cols = st.columns(len(stats))
for idx, sentiment in enumerate(stats['Sentiment']):
    samples = df[df['Sentiment'] == sentiment]['review'].sample(
        n=min(3, len(df[df['Sentiment'] == sentiment])), random_state=1
    ).tolist()
    confidences = df[df['Sentiment'] == sentiment]['Confidence'].sample(
        n=min(3, len(df[df['Sentiment'] == sentiment])), random_state=1
    ).tolist()
    with sample_cols[idx]:
        st.markdown(f"**{sentiment}**")
        for s, c in zip(samples, confidences):
            st.write(f"- {s} *(Confidence: {c:.2f})*")

st.subheader("Test a Sentence")
user_text = st.text_input("Type a sentence to analyze sentiment:")
if user_text:
    with st.spinner("Predicting..."):
        sentiment, score = get_sentiment_hf(user_text)
    st.markdown(f"**Sentiment:** :blue[{sentiment}]")
    st.markdown(f"**Confidence:** :orange[{score:.2f}]")

st.markdown("---")
st.markdown("Made with 🧠 Streamlit & 🤗 Hugging Face Transformers")