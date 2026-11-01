
import streamlit as st
import pandas as pd
import plotly.express as px
from sentiment import analyze_text

st.set_page_config(
    page_title="SentiScope | AI Analytics",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #10111b;
    color: #f5f5f7;
}
[data-testid="stSidebar"] {
    background: #191a29;
}
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #c4a5ff;
}
.subtitle {
    color: #aaaabd;
    font-size: 15px;
}
.result-card {
    background: #202136;
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #373852;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🧠 SentiScope</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-Powered Sentiment & Emotion Intelligence Platform</div>',
    unsafe_allow_html=True
)

st.write("")
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a section",
    ["Text Analysis", "Bulk Analysis", "About"]
)

if page == "Text Analysis":
    st.header("Analyze Your Text")
    text = st.text_area(
        "Enter a review, feedback or any English text",
        placeholder="Example: The camera is amazing but the battery is terrible...",
        height=160
    )

    if st.button("Analyze Sentiment", type="primary"):
        if not text.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("AI is analyzing your text..."):
                try:
                    result = analyze_text(text)

                    st.subheader("Analysis Results")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            "Sentiment",
                            result["sentiment"],
                            f'{result["sentiment_score"]}% confidence'
                        )

                    with col2:
                        st.metric(
                            "Detected Emotion",
                            result["emotion"].title(),
                            f'{result["emotion_score"]}% confidence'
                        )

                    st.subheader("Emotion Distribution")

                    emotion_df = pd.DataFrame(
                        result["all_emotions"]
                    )

                    fig = px.bar(
                        emotion_df,
                        x="label",
                        y="score",
                        color="label",
                        title="Emotion Confidence Scores"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Analysis failed: {e}")

elif page == "Bulk Analysis":
    st.header("Bulk Review Analysis")
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview")
        st.dataframe(df.head())

        column = st.selectbox(
            "Select the column containing reviews",
            df.columns
        )

        if st.button("Analyze All Reviews"):
            results = []

            progress = st.progress(0)

            for i, text in enumerate(df[column].fillna("").astype(str)):
                try:
                    result = analyze_text(text)
                    results.append({
                        "Review": text,
                        "Sentiment": result["sentiment"],
                        "Confidence": result["sentiment_score"],
                        "Emotion": result["emotion"]
                    })
                except Exception:
                    results.append({
                        "Review": text,
                        "Sentiment": "Error",
                        "Confidence": 0,
                        "Emotion": "Unknown"
                    })

                progress.progress((i + 1) / len(df))

            result_df = pd.DataFrame(results)

            st.subheader("Analysis Results")
            st.dataframe(result_df, use_container_width=True)

            st.subheader("Sentiment Distribution")
            counts = result_df["Sentiment"].value_counts().reset_index()
            counts.columns = ["Sentiment", "Count"]

            fig = px.pie(
                counts,
                names="Sentiment",
                values="Count",
                hole=0.45
            )

            st.plotly_chart(fig, use_container_width=True)

            st.download_button(
                "Download Analysis CSV",
                result_df.to_csv(index=False),
                "sentiment_results.csv",
                "text/csv"
            )

elif page == "About":
    st.header("About SentiScope")
    st.write("""
    SentiScope is an NLP-based sentiment and emotion analysis platform.

    **Core Technologies**
    - Python
    - Hugging Face Transformers
    - DistilBERT
    - Streamlit
    - Pandas
    - Plotly

    **Capabilities**
    - Positive and negative sentiment classification
    - Emotion detection
    - Confidence scoring
    - Bulk CSV analysis
    - Interactive visualizations
    """)

st.markdown("---")
st.caption("SentiScope | AI-Powered NLP Analytics")
