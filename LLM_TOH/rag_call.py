# pip install -U langchain langchain-community langchain-core langchain-text-splitters langchain-ollama chromadb beautifulsoup4 tiktoken pypdf lxml matplotlib
import os
import glob
from typing import List, Union, Optional

from langchain_community.document_loaders import WebBaseLoader, BSHTMLLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings 
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# Set User-Agent (recommended to avoid potential blocking and for identification)
os.environ["USER_AGENT"] = "MyRAGSystem/0.1 (your.email@example.com or project-url)"

def load_documents(sources: List[str]) -> List[Document]:
    """
    Load documents from multiple sources, which can be either URLs or local file paths.
    
    Args:
        sources: List of URLs or local file paths
        
    Returns:
        List of loaded documents
    """
    all_documents = []
    
    for source in sources:
        try:
            if source.startswith(('http://', 'https://')):
                # Handle web URLs
                print(f"Loading web source: {source}")
                if source.endswith('.pdf'):
                    # Handle PDF URLs
                    loader = PyPDFLoader(source)
                else:
                    # Handle regular web pages
                    loader = WebBaseLoader(source)
                documents = loader.load()
                print(f"Loaded {len(documents)} document(s) from {source}")
            else:
                # Handle local files
                if os.path.isfile(source):
                    if source.endswith('.html'):
                        print(f"Loading local HTML file: {source}")
                        loader = BSHTMLLoader(source)
                        documents = loader.load()
                    elif source.endswith('.pdf'):
                        print(f"Loading local PDF file: {source}")
                        loader = PyPDFLoader(source)
                        documents = loader.load()
                    else:
                        print(f"Unsupported file type: {source}")
                        continue
                    print(f"Loaded {len(documents)} document(s) from {source}")
                elif os.path.isdir(source):
                    # Handle directory of HTML and PDF files
                    print(f"Loading HTML/PDF files from directory: {source}")
                    html_files = glob.glob(os.path.join(source, "*.html"))
                    pdf_files = glob.glob(os.path.join(source, "*.pdf"))
                    documents = []
                    
                    # Process HTML files
                    for html_file in html_files:
                        print(f"  Loading HTML: {html_file}")
                        loader = BSHTMLLoader(html_file)
                        documents.extend(loader.load())
                    
                    # Process PDF files
                    for pdf_file in pdf_files:
                        print(f"  Loading PDF: {pdf_file}")
                        loader = PyPDFLoader(pdf_file)
                        documents.extend(loader.load())
                        
                    print(f"Loaded {len(documents)} document(s) from directory {source}")
                else:
                    print(f"Unsupported file type or not found: {source}")
                    continue
            
            all_documents.extend(documents)
        except Exception as e:
            print(f"Error loading from {source}: {e}")
    
    return all_documents

def main():
    print("Initializing Retrieval Augmented Generation system...")

    # Define your sources directly in an array
    # Include a mix of local HTML files and web URLs as needed
    sources = [
        "./hanoi_tagged_solution_3_disks.html",
        # "https://www.geeksforgeeks.org/iterative-tower-of-hanoi/"  # Keeping the original source as well
    ]

    print("\nUsing the following document sources:")
    for idx, source in enumerate(sources, 1):
        print(f"{idx}. {source}")
    
    # 1. Load Data
    print("\nLoading documents from sources...")
    data = load_documents(sources)
    if not data:
        print("Error: No data loaded from any sources. Please check your URLs and file paths.")
        return
    print(f"Successfully loaded {len(data)} document(s) in total.")

    # 2. Split Data
    print("Splitting documents into manageable chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_splits = text_splitter.split_documents(data)
    print(f"Split documents into {len(all_splits)} chunks.")

    # 3. Initialize Embeddings
    OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
    print(f"Initializing Ollama embeddings ({OLLAMA_EMBEDDING_MODEL})...")
    try:
        # Ensure you have pulled the nomic-embed-text model in Ollama:
        # ollama pull nomic-embed-text
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
        # Test the embedding model
        print("Testing embedding model...")
        _ = embeddings.embed_query("Test query for embedding model.")
        print(f"Ollama embedding model '{OLLAMA_EMBEDDING_MODEL}' initialized successfully.")
    except Exception as e:
        print(f"Error initializing Ollama embeddings with model '{OLLAMA_EMBEDDING_MODEL}': {e}.")
        print(f"Ensure Ollama is running and you have pulled the '{OLLAMA_EMBEDDING_MODEL}' model (e.g., 'ollama pull {OLLAMA_EMBEDDING_MODEL}').")
        print("Also ensure you have installed 'pip install -U langchain-ollama'.")
        return

    # 4. Create Vector Store
    COLLECTION_NAME = "rag-chroma-nomic-embed"
    print(f"Creating ChromaDB vector store (in-memory, collection: {COLLECTION_NAME})...")
    try:
        vectorstore = Chroma.from_documents(
            documents=all_splits,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
        )
        retriever = vectorstore.as_retriever()
        print("Vector store created successfully.")
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return

    # 5. Setup Ollama LLM for Chat
    OLLAMA_CHAT_MODEL = 'gemma3:4b' # Or your preferred Ollama chat model like 'llama3', 'mistral'
    print(f"Using Ollama chat model: {OLLAMA_CHAT_MODEL}")
    try:
        model_local = ChatOllama(model=OLLAMA_CHAT_MODEL)
        # Perform a quick test invocation to check connectivity
        print(f"Testing Ollama chat model '{OLLAMA_CHAT_MODEL}' connection...")
        _ = model_local.invoke("This is a test prompt.")
        print(f"Ollama chat model '{OLLAMA_CHAT_MODEL}' connection successful.")
    except Exception as e:
        print(f"Error connecting to Ollama chat model '{OLLAMA_CHAT_MODEL}': {e}")
        print(f"Please ensure Ollama is running and the specified model is available (e.g., 'ollama pull {OLLAMA_CHAT_MODEL}').")
        print("Also ensure you have installed 'pip install -U langchain-ollama'.")
        return

    # 6. Define Prompt Template
    # template = """Answer the question based *only* on the following context. If the context does not contain the answer, state that you do not have enough information from the provided context to answer. Do not make up information.

    # Context:
    # {context}

    # Question: {question}

    # Answer:
    # """
    template = """Answer the question based on the following context if relevant. If the context does not contain information related to the question, you may use your general knowledge to provide an answer.

    Context:
    {context}

    Question: {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 7. Define RAG Chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model_local
        | StrOutputParser()
    )
    print("RAG chain is configured and ready.")
    print("\n--- Starting Question Answering Session ---")
    print("Type 'quit', 'exit', or 'q' to end the session.")

    # 8. Continuous Question Answering Loop
    while True:
        user_question = input("\nEnter your question: ")
        if user_question.lower() in ["quit", "exit", "q"]:
            print("Exiting RAG system.")
            break
        if not user_question.strip():
            print("No question entered. Please provide a question.")
            continue

        print("Processing your question...")
        try:
            answer = chain.invoke(user_question)
            print("\nAnswer:")
            print(answer)
        except Exception as e:
            print(f"Error during RAG chain invocation: {e}")

if __name__ == "__main__":
    # --- Pre-run Checklist ---
    # 1. Ensure Ollama is installed and running.
    # 2. Pull the necessary Ollama models:
    #    ollama pull nomic-embed-text
    #    ollama pull gemma3:4b (or your chosen OLLAMA_CHAT_MODEL)
    # 3. Ensure Python packages listed in requirements.txt (or below) are installed.
    main()