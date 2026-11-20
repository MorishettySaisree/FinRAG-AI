from langchain_ollama import ChatOllama


def agentic_chunk(text):

    # Load local LLM (offline)
    llm = ChatOllama(
        model="llama2",
        temperature=0
    )

    # Prompt for chunking
    prompt = f"""
You are a text chunking expert.

Split the given text into logical chunks.

Rules:
1. Each chunk should be around 200-300 words
2. Split at natural topic boundaries
3. Keep related information together
4. Put <<<SPLIT>>> between chunks
5. Do not remove any text
6. Return only text with <<<SPLIT>>> markers

Text:
{text}

Return the text with <<<SPLIT>>> markers:
"""

    # Ask LLM to chunk
    response = llm.invoke(prompt)

    marked_text = response.content

    # Split chunks
    chunks = marked_text.split("<<<SPLIT>>>")

    # Clean chunks
    clean_chunks = []

    for chunk in chunks:
        cleaned = chunk.strip()

        if cleaned:
            clean_chunks.append(cleaned)

    return clean_chunks


# Test independently
if __name__ == "__main__":

    sample_text = """
    Retrieval Augmented Generation improves language models by retrieving external information.

    Vector databases store embeddings generated from documents.

    Similarity search identifies the most relevant context for a query.

    The retrieved context is combined with the user query.

    The language model generates an answer using augmented context.

    Multimodal systems process text, PDF, DOCX, images and audio together.
    """

    chunks = agentic_chunk(sample_text)

    print("AGENTIC CHUNKING RESULTS")
    print("=" * 50)

    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}")
        print(chunk)
        print("-" * 50)