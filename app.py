import streamlit as st
from llm_local import load_llm, analyze_article

st.set_page_config(page_title="Offline Fake News Detector")

st.title("📰 Offline Fake News Detector")

st.write("Fully local, privacy-focused fake news detection system.")

@st.cache_resource
def get_model():
    return load_llm()

llm = get_model()

article = st.text_area("Paste article text", height=300)

if st.button("Analyze"):

    if not article.strip():
        st.warning("Please enter article text.")
    else:
        with st.spinner("Analyzing with Mistral 7B..."):
            result = analyze_article(article, llm)

        st.subheader("Verdict")
        st.write(f"**{result['verdict']}**")

        st.subheader("Confidence")
        st.write(f"{result['confidence']}%")

        st.subheader("Key Claims")
        for claim in result["key_claims"]:
            st.write(f"- {claim}")

        st.subheader("Explanation")
        st.write(result["explanation"])