from langchain_text_splitters import RecursiveCharacterTextSplitter


def recursive_chunk(text):

    recursive_splitter = RecursiveCharacterTextSplitter(

        # try preserving structure first
        separators=[
            "\n\n",     # paragraph
            "\n",       # line
            ". ",       # sentence
            " ",        # word
            ""          # character fallback
        ],

        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = recursive_splitter.split_text(text)

    return chunks


# Test independently
if __name__ == "__main__":

    sample_text = """
    Retrieval Augmented Generation improves language models by retrieving external knowledge.

    Vector databases store embeddings generated from documents.

    Similarity search identifies relevant information for a query.

    The language model uses retrieved context to generate accurate responses.

    Multimodal systems can process text, PDF, images, audio and DOCX files.
    """

    chunks = recursive_chunk(sample_text)

    print("RECURSIVE CHUNKING RESULTS")
    print("=" * 50)

    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}")
        print(chunk)
        print("-" * 50)