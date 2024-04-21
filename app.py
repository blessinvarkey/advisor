import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain.chains.question_answering import load_qa_chain

# Accessing the API key securely
api_key = st.secrets["OPENAI_API_KEY"]

# Use the API key directly where needed (e.g., initializing OpenAI)
openai_client = OpenAI(api_key=api_key)
embeddings = OpenAIEmbeddings(api_key=api_key)

# Continue with your application logic
raw_text = st.secrets["RAW_TEXT"]
texts = [text.strip() for text in raw_text.split('\n') if text.strip()]
document_search = FAISS.from_texts(texts, embeddings)
chain = load_qa_chain(openai_client, chain_type="generic")

# Streamlit interface to run queries
st.title("Query Assistant")
query = st.text_input("Enter your question:")
if query:
    docs = document_search.similarity_search(query)
    result = chain.run(input_documents=docs, question=query)
    st.write("Answer:", result)
