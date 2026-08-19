import os
import sys
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

BATCH_SIZE = 20
BATCH_DELAY = 10
MAX_RETRIES = 5

def add_batch_with_retry(vectorstore, batch, max_retries=MAX_RETRIES):
    for attempt in range(max_retries):
        try:
            vectorstore.add_documents(batch)
            return
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) and attempt < max_retries - 1:
                print(f"    Rate limited - waiting 60s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(60)
            else:
                raise

def build_vector_store(ticker: str):
    print(f"Processing 10-K for {ticker}...")

    pdf_path = f"data/{ticker.upper()}_10K.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: Could not find {pdf_path}. Make sure it is saved in the data/ folder")
        return

    print("1. Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"    -> Loaded {len(docs)} pages.")

    print("2. Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)
    print(f"    -> Split into {len(chunks)} chunks.")

    print("3. Generating embeddings and saving to ChromaDB...")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    persist_directory = f"chroma_db/{ticker.upper()}"

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]        
        add_batch_with_retry(vectorstore, batch)
        print(f"    -> Embedded {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")
        time.sleep(BATCH_DELAY)

    print(f"Successfully built vector store for {ticker.upper()} at {persist_directory}/\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        build_vector_store(sys.argv[1])
    else:
        print("Please provide a ticker symbol as an argument.")