import streamlit as st
from main import retriever, llm  # імпорт з основного файлу

st.title("CyberMentorAI")
query = st.text_input("Введи запит:")

if query:
    docs = retriever.get_relevant_documents(query)
    combined_text = "\n\n".join([doc.page_content for doc in docs])
    limited_text = combined_text[:1000]

    prompt = f"Поясни: {query}. Використай інформацію: {limited_text}"
    result = llm.invoke(prompt)

    st.subheader("Відповідь:")
    st.write(result)

    st.subheader("Джерела:")
    for i, doc in enumerate(docs):
        page = doc.metadata.get('page', '?')
        snippet = doc.page_content[:150]
        st.write(f"[{i+1}] Сторінка {page}: {snippet}...")
