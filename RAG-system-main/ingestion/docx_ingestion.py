import os
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def load_documents(docx_path="../data/docx"):
    print(f"Loading DOCX documents from {docx_path}...")

    if not os.path.exists(docx_path):
        raise FileNotFoundError(f"Directory '{docx_path}' does not exist.")

    documents = []

    for file in os.listdir(docx_path):
        if file.endswith(".docx"):
            loader = Docx2txtLoader(os.path.join(docx_path, file))
            documents.extend(loader.load())

    if len(documents) == 0:
        raise FileNotFoundError("No DOCX files found.")
    for doc in documents:
        doc.metadata["modality"] = "docx"

    print(f"Loaded {len(documents)} documents")


    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    print("\nSplitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks


def create_vector_store(chunks, persist_directory):
    print("\nCreating vector store with Chroma...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print(f"Vector store created and persisted at '{persist_directory}'.")
    return vector_store


def main():
    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    DOCX_PATH = os.path.join(BASE_DIR, "data", "docx")
    DB_PATH = os.path.join(BASE_DIR, "db", "chroma_db")

    documents = load_documents(docx_path=DOCX_PATH)

    chunks = split_documents(documents)

    vector_store = create_vector_store(chunks,DB_PATH)


if __name__ == "__main__":
    main()
