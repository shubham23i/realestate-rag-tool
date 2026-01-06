import streamlit as st
from rag import process_urls, generate_answer

st.set_page_config(page_title="Real Estate RAG Tool", layout="wide")

st.title("🏡 Real Estate Research Tool")

# -------- Sidebar --------
st.sidebar.header("📥 Enter URLs")

url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

process_btn = st.sidebar.button("🚀 Process URLs")

status = st.empty()

# -------- Process URLs --------
if process_btn:
    urls = [u for u in (url1, url2, url3) if u.strip()]

    if not urls:
        status.error("Please enter at least one URL")
    else:
        for msg in process_urls(urls):
            status.info(msg)
        status.success("URLs processed successfully!")

# -------- Question --------
st.divider()
query = st.text_input("❓ Ask a question")

if query:
    try:
        answer, sources = generate_answer(query)

        st.subheader("🧠 Answer")
        st.write(answer)

        if sources:
            st.subheader("🔗 Sources")
            for s in sources:
                st.write(s)

    except RuntimeError:
        st.error("Please process URLs first.")
