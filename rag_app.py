import os
import tempfile
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS

load_dotenv()

st.title("SmartPDF RAG Assistant")

uploaded_files = st.file_uploader(
    "Upload PDF Files",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    all_chunks = []

    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:
            tmp.write(file.getvalue())
            temp_pdf = tmp.name

        loader = PyPDFLoader(temp_pdf)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        split_docs = splitter.split_documents(docs)

        for doc in split_docs:
            doc.metadata["pdf_name"] = file.name

        all_chunks.extend(split_docs)

    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_documents(
        all_chunks,
        embeddings
    )

    st.success(
        f"Indexed {len(all_chunks)} chunks"
    )

    question = st.text_input(
        "Ask a Question"
    )

    if question:

        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )

        results = retriever.invoke(question)

        context = "\n\n".join(
            [doc.page_content for doc in results]
        )

        llm = ChatOpenAI(
            model="gpt-3.5-turbo"
        )

        prompt = f"""
        Answer using only the context below.

        Context:
        {context}

        Question:
        {question}
        """

        response = llm.invoke(prompt)

        st.subheader("Answer")

        st.write(response.content)
        
        st.subheader("Sources")

        for doc in results:
          st.write(
        f"{doc.metadata.get('pdf_name')} - Page {doc.metadata.get('page', 0) + 1}"
    )