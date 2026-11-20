
from retrieval.retriever import db

TEST_CASES = [
    {
        "question": "What was the company's revenue in 2025?",
        "keywords": ["15,00,000", "revenue"],
        "expected_answerable": True
    },
    {
        "question": "What was the net profit in 2024?",
        "keywords": ["2,40,000", "profit"],
        "expected_answerable": True
    },
    {
        "question": "How much did net profit increase from 2024 to 2025?",
        "keywords": ["90,000", "profit"],
        "expected_answerable": True
    },
    {
        "question": "What were the operating expenses in 2025?",
        "keywords": ["9,50,000", "expenses"],
        "expected_answerable": True
    },
    {
        "question": "What was the company's revenue in 2023?",
        "keywords": ["revenue", "2023"],
        "expected_answerable": False
    }
]

K = 5
THRESHOLD = 0.55


def evaluate_retrieval():
    passed = 0
    answerable_cases = 0
    successful_hits = 0

    print("\n===== FINRAG AI RETRIEVAL EVALUATION =====")

    for index, test in enumerate(TEST_CASES, start=1):
        question = test["question"]
        expected_answerable = test["expected_answerable"]

        results = db.similarity_search_with_score(
            query=question,
            k=K
        )

        relevant_results = [
            (doc, score)
            for doc, score in results
            if score < THRESHOLD
        ]

        combined_text = " ".join(
            doc.page_content.lower()
            for doc, _ in relevant_results
        )

        # Simple keyword-based relevance check.
        if expected_answerable:
            hit = all(
                keyword.lower() in combined_text
                for keyword in test["keywords"]
            )
        else:
            # The requested year-specific revenue fact should
            # not be supported by the sample report.
            hit = (
                "revenue" in combined_text
                and "2023" in combined_text
                and "15,00,000" in combined_text
            )

        passed += int(hit)

        if expected_answerable:
            answerable_cases += 1
            successful_hits += int(hit)

        print(f"\nTest {index}: {question}")
        print(f"Expected answerable: {expected_answerable}")
        print(f"Documents above threshold: {len(relevant_results)}")
        print(f"Keyword check passed: {hit}")

        for doc, score in relevant_results:
            print(f"  Distance: {score:.4f}")
            print(f"  Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"  Content: {doc.page_content[:250]!r}")

    total = len(TEST_CASES)

    print("\n===== SUMMARY =====")
    print(f"Keyword checks passed: {passed}/{total}")
    print(f"Overall keyword-check rate: {passed / total:.2%}")

    if answerable_cases:
        print(
            "Answerable-question hit rate: "
            f"{successful_hits / answerable_cases:.2%}"
        )

    print(
        "\nNote: These are simple keyword-based checks, "
        "not a complete semantic relevance or answer-quality evaluation."
    )


if __name__ == "__main__":
    evaluate_retrieval()
