import streamlit as st
from rag import process_urls, generate_answer

st.set_page_config(page_title="Real Estate RAG Tool", layout="wide")

st.title("🏡 Real Estate Research Tool")

# -------------------- SIDEBAR --------------------
st.sidebar.header("📥 Input URLs")

url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

process_url_button = st.sidebar.button("🚀 Process URLs")

status_box = st.empty()

# -------------------- PROCESS URLS --------------------
if process_url_button:
    urls = [url for url in (url1, url2, url3) if url.strip()]

    if not urls:
        status_box.error("❌ Please provide at least one valid URL")
    else:
        for status in process_urls(urls):
            status_box.info(status)

        status_box.success("✅ URLs processed successfully!")

# -------------------- QUESTION INPUT --------------------
st.divider()
query = st.text_input("❓ Ask a question based on the processed content")

# -------------------- GENERATE ANSWER --------------------
if query:
    try:
        answer, sources = generate_answer(query)

        st.subheader("🧠 Answer")
        st.write(answer)

        if sources:
            st.subheader("🔗 Sources")
            for source in sources:
                st.write(source)

    except RuntimeError:
        st.error("⚠️ Please process URLs before asking a question.")
