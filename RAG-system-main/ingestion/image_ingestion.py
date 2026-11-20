import os
import pytesseract
from PIL import Image
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Windows only if needed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"



def load_documents(image_path="../data/images"):
    print(f"Loading Images from {image_path}...")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Directory '{image_path}' does not exist.")

    documents = []

    for file in os.listdir(image_path):

        if file.endswith((".png", ".jpg", ".jpeg")):

            image = Image.open(
                os.path.join(image_path, file)
            )

            text = pytesseract.image_to_string(image)

            doc = Document(
                page_content=text,
                metadata={"source": file, "modality": "image"}
            )

            documents.append(doc)

    if len(documents) == 0:
        raise FileNotFoundError("No image files found.")

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
    print("Starting embedding + DB insertion...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print("DB insertion complete")
    print(f"Vector store created and persisted at '{persist_directory}'.")
    return vector_store


def main():
    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    IMAGE_PATH = os.path.join(BASE_DIR, "data", "images")
    DB_PATH = os.path.join(BASE_DIR, "db", "chroma_db")

    documents = load_documents(image_path=IMAGE_PATH)

    chunks = split_documents(documents)

    vector_store = create_vector_store(
        chunks,DB_PATH
    )


if __name__ == "__main__":
    main()