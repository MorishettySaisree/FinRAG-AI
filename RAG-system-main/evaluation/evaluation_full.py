
import time
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from retrieval.retriever import db
from generation.answer_generation import generate_answer


K = 5

TEST_CASES = [
    {
        "question": "What was the company's revenue in 2025?",
        "keywords": ["15,00,000", "revenue"],
        "answerable": True,
    },
    {
        "question": "What was the net profit in 2024?",
        "keywords": ["2,40,000", "profit"],
        "answerable": True,
    },
    {
        "question": "How much did net profit increase from 2024 to 2025?",
        "keywords": ["90,000", "profit"],
        "answerable": True,
    },
    {
        "question": "What were the operating expenses in 2025?",
        "keywords": ["9,50,000", "expenses"],
        "answerable": True,
    },
    {
        "question": "What was the company's revenue in 2023?",
        "keywords": [],
        "answerable": False,
    },
]


def contains_keywords(text, keywords):
    text = text.lower().replace(",", "").replace(" ", "")
    return all(
        keyword.lower().replace(",", "").replace(" ", "") in text
        for keyword in keywords
    )


def evaluate():
    recall_hits = 0
    reciprocal_ranks = []
    correct_answers = 0
    supported_answers = 0
    abstentions = 0
    unanswerable_count = 0
    latencies = []

    print("\n===== FINRAG AI FULL EVALUATION =====")

    for index, test in enumerate(TEST_CASES, start=1):
        question = test["question"]
        keywords = test["keywords"]
        answerable = test["answerable"]

        start = time.perf_counter()

        # Retrieve the top K chunks from the existing database.
        results = db.similarity_search_with_score(
            query=question,
            k=K,
        )

        docs = [doc for doc, score in results]

        # Retrieval relevance: first chunk containing all gold keywords.
        rank = None

        if answerable:
            for position, doc in enumerate(docs, start=1):
                if contains_keywords(doc.page_content, keywords):
                    rank = position
                    break

            if rank is not None:
                recall_hits += 1
                reciprocal_ranks.append(1 / rank)
            else:
                reciprocal_ranks.append(0)

        context = "\n".join(doc.page_content for doc in docs)

        # Generate an answer using the existing local LLM.
        answer = generate_answer(
            query=question,
            relevant_docs=docs,
            chat_history=None,
        )

        latency = time.perf_counter() - start
        latencies.append(latency)

        is_abstention = (
            answer.strip().lower() == "information not found."
        )

        if answerable:
            is_correct = (
                not is_abstention
                and contains_keywords(answer, keywords)
            )

            if is_correct:
                correct_answers += 1

            # Heuristic grounding check:
            # Does the retrieved context contain the expected fact?
            if contains_keywords(context, keywords):
                supported_answers += 1

        else:
            unanswerable_count += 1
            if is_abstention:
                abstentions += 1

        print(f"\nTest {index}: {question}")
        print(f"Answerable: {answerable}")
        print(f"Answer: {answer}")
        print(f"Latency: {latency:.2f} seconds")

        if answerable:
            print(f"Relevant chunk rank: {rank}")
            print(f"Answer correct by keyword check: {is_correct}")
            print(
                "Expected fact present in retrieved context:",
                contains_keywords(context, keywords),
            )
        else:
            print(f"Correctly abstained: {is_abstention}")

    answerable_count = sum(
        test["answerable"] for test in TEST_CASES
    )

    total = len(TEST_CASES)

    recall_at_5 = recall_hits / answerable_count
    mrr = sum(reciprocal_ranks) / answerable_count
    accuracy = correct_answers / answerable_count
    groundedness_proxy = supported_answers / answerable_count
    abstention_rate = (
        abstentions / unanswerable_count
        if unanswerable_count else 0
    )
    average_latency = sum(latencies) / len(latencies)

    print("\n===== EVALUATION SUMMARY =====")
    print(f"Total questions: {total}")
    print(f"Answerable questions: {answerable_count}")
    print(f"Recall@5: {recall_at_5:.2%}")
    print(f"MRR: {mrr:.4f}")
    print(f"Answer accuracy: {accuracy:.2%}")
    print(f"Grounding evidence coverage (proxy): {groundedness_proxy:.2%}")
    print(f"Correct abstention rate: {abstention_rate:.2%}")
    print(f"Average end-to-end latency: {average_latency:.2f} seconds")

    print(
        "\nNote: These are preliminary results on a small test set. "
        "Keyword-based accuracy and grounding coverage are proxies, "
        "not complete semantic evaluations."
    )


if __name__ == "__main__":
    evaluate()
