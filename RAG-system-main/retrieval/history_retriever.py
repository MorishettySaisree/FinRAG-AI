import os
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Multiple vector DBs
text_db = Chroma(
    persist_directory=os.path.join(BASE_DIR, "db", "text_db"),
    embedding_function=embeddings
)

pdf_db = Chroma(
    persist_directory=os.path.join(BASE_DIR, "db", "pdf_db"),
    embedding_function=embeddings
)

docx_db = Chroma(
    persist_directory=os.path.join(BASE_DIR, "db", "docx_db"),
    embedding_function=embeddings
)

image_db = Chroma(
    persist_directory=os.path.join(BASE_DIR, "db", "image_db"),
    embedding_function=embeddings
)

audio_db = Chroma(
    persist_directory=os.path.join(BASE_DIR, "db", "audio_db"),
    embedding_function=embeddings
)

# Local LLM
model = ChatOllama(
    model="llama2"
)

# Chat memory
chat_history = []


# Retrieve docs from all DBs
def retrieve_docs(db, query, k=1, threshold=0.60):

    docs = []

    results = db.similarity_search_with_score(query, k=k)

    for doc, score in results:

        if score < threshold:
            docs.append(doc)

    return docs


def ask_question(user_question):

    print(f"\nYou asked: {user_question}")

    # Rewrite follow-up question
    if chat_history:

        messages = [
            SystemMessage(
                content="Rewrite the new question into a standalone searchable question using chat history. Return only rewritten question."
            )
        ] + chat_history + [
            HumanMessage(content=user_question)
        ]

        result = model.invoke(messages)

        search_question = result.content.strip()

        print("Searching for:", search_question)

    else:
        search_question = user_question

    # Search all DBs
    docs = (
        retrieve_docs(text_db, search_question)
        + retrieve_docs(pdf_db, search_question)
        + retrieve_docs(docx_db, search_question)
        + retrieve_docs(image_db, search_question)
        + retrieve_docs(audio_db, search_question)
    )

    # No results
    if not docs:
        print("Information not found")
        return

    # Show retrieved docs
    print("\nRetrieved Documents")

    for i, doc in enumerate(docs, 1):

        print(f"\nDocument {i}")
        print("Metadata:", doc.metadata)
        print(doc.page_content[:300])

    # Build context
    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    # Prompt
    prompt = f"""
Answer ONLY using the provided context.

Context:
{context}

Question:
{user_question}

Rules:
1. Use only context
2. Do not make up information
3. If answer missing say Information not found

Answer:
"""

    result = model.invoke(prompt)

    answer = result.content

    # Save history
    chat_history.append(
        HumanMessage(content=user_question)
    )

    chat_history.append(
        AIMessage(content=answer)
    )

    print("\nAnswer:")
    print(answer)


def start_chat():

    print("Conversation RAG started (type quit to exit)")

    while True:

        question = input("\nYour question: ")

        if question.lower() == "quit":
            print("Exiting chat. Goodbye!")
            break

        ask_question(question)


if __name__ == "__main__":
    start_chat()
