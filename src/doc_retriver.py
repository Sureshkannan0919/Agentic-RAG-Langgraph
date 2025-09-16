from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import pandas as pd
import pickle
import os
from langchain_core.tools import tool

# Configuration
VECTOR_DB_PATH = "chroma_db"
DOCS_PICKLE_PATH = "docs_list.pkl"
CSV_PATH = "/home/suresh/projects/langchain_data.csv"

def load_or_create_vectorstore():
    """Load existing vectorstore or create a new one"""
    
    # Check if vector database already exists
    if os.path.exists(VECTOR_DB_PATH):
        print("Loading existing vector database...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma(
            collection_name="langchain-docs-chroma",
            embedding_function=embeddings,
            persist_directory=VECTOR_DB_PATH
        )
        return vectorstore
    
    # Create new vector database
    print("Creating new vector database...")
    
    # Load CSV data
    df = pd.read_csv(CSV_PATH)
    
    # Load or create documents
    try:
        with open(DOCS_PICKLE_PATH, "rb") as f:
            docs_list = pickle.load(f)
        if docs_list:
            print("Documents loaded from pickle file.")
    except Exception as e:
        print("Error loading documents:", e)
        print("Loading documents from URLs...")
        docs = [WebBaseLoader(url).load() for url in df["URL"]]
        docs_list = [item for sublist in docs for item in sublist]
        
        with open(DOCS_PICKLE_PATH, "wb") as f:
            pickle.dump(docs_list, f)
        print("Documents saved to pickle file.")
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, 
        chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create and persist vector store
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="langchain-docs-chroma",
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )
    
    print(f"Vector database created and saved to {VECTOR_DB_PATH}")
    return vectorstore

# Load or create the vector store
vectorstore = load_or_create_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

@tool
def langchain_docs_retriever(question: str) -> str:
    """Retrieves information from the Langchain documentation."""
    docs = retriever.get_relevant_documents(question)
    str_docs = ""
    for doc in docs:
        str_docs += "\n\n" + doc.page_content
    return str_docs

# # Test the retriever
# if __name__ == "__main__":
#     output = langchain_docs_retriever("how to do memmory management in lagchain?")
#     print(output)

