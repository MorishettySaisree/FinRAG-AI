import os
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg-8.1.1-essentials_build\bin"
import whisper
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def load_documents(audio_path="../data/audio"):
    print(f"Loading Audio files from {audio_path}...")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Directory '{audio_path}' does not exist.")

    model = whisper.load_model("base")

    documents = []

    for file in os.listdir(audio_path):

        if file.endswith((".mp3", ".wav")):

            result = model.transcribe(
                os.path.join(audio_path, file)
            )

            text = result["text"]

            doc = Document(
                page_content=text,
                metadata={
                    "source": file,
                    "modality": "audio"
                }
            )

            documents.append(doc)

    if len(documents) == 0:
        raise FileNotFoundError("No audio files found.")

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

    print("Embeddings model loaded")

    print("Starting DB insertion...")

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

    AUDIO_PATH = os.path.join(BASE_DIR, "data", "audio")
    DB_PATH = os.path.join(BASE_DIR, "db", "chroma_db")

    documents = load_documents(audio_path=AUDIO_PATH)

    chunks = split_documents(documents)

    vector_store = create_vector_store(
        chunks,DB_PATH
    )


if __name__ == "__main__":
    main()