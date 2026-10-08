# Multi-PDF RAG with LangChain

A Streamlit app that lets you upload multiple PDF files, indexes their contents with LangChain and FAISS, and answers questions using OpenAI. Answers include the source PDF and page for each retrieved passage.

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and set `OPENAI_API_KEY` to your OpenAI API key.
4. Start the app:

   ```bash
   streamlit run rag_app.py
   ```

5. Upload one or more PDF files and ask a question about their contents.

The PDFs are uploaded through the app and do not need to be committed to this repository. Keep `.env` and API keys out of source control.
