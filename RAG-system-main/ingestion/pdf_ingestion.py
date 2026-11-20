import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def load_documents(pdf_path="../data/pdfs"):
    print(f"Loading PDF documents from {pdf_path}...")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Directory '{pdf_path}' does not exist.")

    documents = []

    for file in os.listdir(pdf_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_path, file))
            documents.extend(loader.load())

    if len(documents) == 0:
        raise FileNotFoundError("No PDF files found.")
    for doc in documents:
        doc.metadata["modality"] = "pdf"

    print(f"Loaded {len(documents)} documents")

    return documents


def split_documents(documents, chunk_size=500, chunk_overlap=100):
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

    print("---Creating vector store---")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print("---Finished creating vector store---")
    return vector_store


def main():
    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    PDF_PATH = os.path.join(BASE_DIR, "data", "pdfs")
    DB_PATH = os.path.join(BASE_DIR, "db", "chroma_db")

    documents = load_documents(pdf_path=PDF_PATH)

    chunks = split_documents(documents)

    vector_store = create_vector_store(chunks,DB_PATH)


if __name__ == "__main__":
    main()
