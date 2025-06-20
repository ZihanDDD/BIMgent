import os
from typing import List

from openai import OpenAI
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma  # Updated import
from langchain_community.document_loaders import TextLoader

###############################################################################
# Configuration                                                               #
###############################################################################
INPUT_FOLDER: str = "./res/vectorworks/builders"
PERSIST_DIR: str = "./chroma_db"           # keep DB files separate from code
BATCH_SIZE: int = 50                        # tune for your memory constraints

###############################################################################
# Utility helpers                                                             #
###############################################################################

def _get_markdown_documents(folder: str) -> List[str]:
    """Return a list of LangChain Document objects for every .md file found."""
    docs = []
    for file in os.listdir(folder):
        if file.lower().endswith(".md"):
            loader = TextLoader(os.path.join(folder, file), encoding="utf-8")
            docs.extend(loader.load())
    return docs


def _create_embeddings(api_key: str):
    """Create a reusable embeddings object."""
    return OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=768,
        api_key=api_key,
    )

###############################################################################
# Ingestion + Persistence                                                     #
###############################################################################

def ingest_documents(
    input_folder: str = INPUT_FOLDER,
    persist_dir: str = PERSIST_DIR,
    batch_size: int = BATCH_SIZE,
):
    """Read markdown files, embed them, and persist to a local Chroma DB.

    Calling this repeatedly will *append* only NEW docs if the DB already exists.
    """
    
    load_dotenv()
    api_key = os.getenv("OA_OPENAI_KEY")
    if not api_key:
        raise ValueError("Environment variable OA_OPENAI_KEY not set.")
    docs = _get_markdown_documents(input_folder)
    if not docs:
        raise FileNotFoundError(f"No .md files found under {input_folder}")

    embeddings = _create_embeddings(api_key)

    # Initialise (or open) the collection once, then batch-add docs
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    current_batch = []

    for doc in docs:
        current_batch.append(doc)
        if len(current_batch) >= batch_size:
            db.add_documents(current_batch)
            current_batch.clear()

    # Flush any leftover docs < batch_size
    if current_batch:
        db.add_documents(current_batch)
    
    # No need to call persist() - Chroma automatically persists documents now
    print(f"Ingested {len(docs)} documents into '{persist_dir}'.")
    return db

###############################################################################
# Vector store loading                                                        #
###############################################################################

def load_vectorstore(api_key: str, persist_dir: str = PERSIST_DIR):
    """Load the persisted Chroma DB; raise if not present."""
    if not os.path.isdir(persist_dir) or not os.listdir(persist_dir):
        raise FileNotFoundError(
            f"No vector store found in '{persist_dir}'. Run ingest_documents() first."
        )
    embeddings = _create_embeddings(api_key)
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

###############################################################################
# Quick CLI demo                                                              #
###############################################################################

def query_builder(query):
    load_dotenv()
    api_key = os.getenv("OA_OPENAI_KEY")
    if not api_key:
        raise ValueError("Environment variable OA_OPENAI_KEY not set.")

    # ---------------------------------------------------------------------
    # Step 1: Ensure the vector DB exists (auto‑ingest on first run)
    # ---------------------------------------------------------------------
    try:
        db = load_vectorstore(api_key)
    except FileNotFoundError:
        print("Vector store not found – ingesting documents now…")
        db = ingest_documents()  # No need to pass api_key here as it loads from env

    # ---------------------------------------------------------------------
    # Step 2: Query the DB
    # ---------------------------------------------------------------------
    retriever = db.as_retriever(search_kwargs={"k": 1})
    docs = retriever.invoke(query)
    
    # example: just want the plain text
    docs_str = "\n\n".join(d.page_content for d in docs)

    if not docs:
        print("No documents matched the query.")
        
    
    return docs_str