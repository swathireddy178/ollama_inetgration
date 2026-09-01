import streamlit as st

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

st.title("🤖 Website RAG Chatbot")
st.write("Ask questions about LangSmith documentation")

@st.cache_resource
def setup_rag():
    
    loader = WebBaseLoader(
        "https://docs.langchain.com/langsmith/administration-overview"
    )

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    split_docs = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(
        model="embeddinggemma"
    )

    vectorstoredb = FAISS.from_documents(
        split_docs,
        embeddings
    )

    retriever = vectorstoredb.as_retriever()

    llm = Ollama(model="llama3")

    prompt = ChatPromptTemplate.from_template(
        """
Answer the following question based only on the provided context:

<context>
{context}
</context>

Question:
{input}
"""
    )

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    return retrieval_chain

chain = setup_rag()

question = st.text_input("Ask a question")

if st.button("Get Answer"):

    with st.spinner("Thinking..."):

        response = chain.invoke(
            {"input": question}
        )

        st.success(response["answer"])