import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def load_documents(docs_path="../data/texts"):
    print(f"Loading documents from {docs_path}...")
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Directory '{docs_path}' does not exist.")
    loader = DirectoryLoader(path=docs_path, glob="*.txt", loader_cls=TextLoader,loader_kwargs={"encoding": "utf-8"})
    documents = loader.load()


    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in directory '{docs_path}'.")

    for doc in documents:
        doc.metadata["modality"] = "text"

    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    print(f"\nSplitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks, persist_directory):
    print(f"\nCreating vector store with Chroma...")
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("---Creating vector store---")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("---Finished creating vector store---")
    print(f"Vector store created and persisted at '{persist_directory}'.")
    return vector_store

def main():
    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    DOCS_PATH = os.path.join(
        BASE_DIR,
        "data",
        "texts"
    )

    DB_PATH = os.path.join(
        BASE_DIR,
        "db",
        "chroma_db"
    )

    # Load documents
    documents = load_documents(docs_path=DOCS_PATH)

    # Split documents into chunks
    chunks = split_documents(documents)

    # Create vector store and persist it
    vector_store = create_vector_store(chunks, DB_PATH)
if __name__ == '__main__':
    main()
