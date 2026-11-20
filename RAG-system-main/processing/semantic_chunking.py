from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings


def semantic_chunk(text):

    # Same embedding model used in whole project
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Semantic chunking based on meaning
    semantic_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=70
    )

    # Split text semantically
    chunks = semantic_splitter.split_text(text)

    return chunks


# Test file independently
if __name__ == "__main__":

    sample_text = """
    Retrieval Augmented Generation improves language models by retrieving external knowledge.
    Vector databases store embeddings generated from documents.
    Similarity search identifies relevant information for a given query.
    The language model uses retrieved context to generate more accurate responses.
    Multimodal RAG can process text, PDF, images, audio and document files.
    """

    chunks = semantic_chunk(sample_text)

    print("SEMANTIC CHUNKING RESULTS")
    print("=" * 50)

    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}")
        print(chunk)
        print("-" * 50)