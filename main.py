import streamlit as st
from rag import process_urls, generate_answer

st.set_page_config(page_title="Real Estate Research Tool")

st.title("Real Estate Research Tool")

url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

status = st.empty()

if st.sidebar.button("Process URLs"):
    urls = [u for u in (url1, url2, url3) if u.strip()]
    if not urls:
        status.warning("Please enter at least one URL")
    else:
        for msg in process_urls(urls):
            status.info(msg)

query = st.text_input("Ask a question")
if query:
    answer = generate_answer(query)
    st.subheader("Answer")
    st.write(answer)
