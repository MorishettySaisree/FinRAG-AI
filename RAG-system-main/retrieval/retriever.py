
import os
import re
from collections import defaultdict

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma


# --------------------------------------------------
# DATABASE CONFIGURATION
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DB_PATH = os.path.join(BASE_DIR, "db", "chroma_db")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"local_files_only": True},
)

db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"},
)


# --------------------------------------------------
# GENERAL HELPERS
# --------------------------------------------------

def normalize(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def get_company_identifier(metadata):
    company = (
        metadata.get("company")
        or metadata.get("company_name")
    )

    if company:
        return str(company).strip()

    source = str(metadata.get("source", ""))
    filename = source.replace("\\", "/").split("/")[-1]
    filename = os.path.splitext(filename)[0]

    filename = re.sub(
        r"^FinRAG_(?:Sample|TXT)_\d+_",
        "",
        filename,
        flags=re.IGNORECASE,
    )

    return filename.replace("_", " ").strip()


def extract_named_entities(query):
    """
    Detect likely proper names in a question.

    Examples:
    What was AsterNova Technologies' revenue?
    -> AsterNova Technologies

    Where is NoblePeak Chemicals headquartered?
    -> NoblePeak Chemicals
    """

    ignored = {
        "what", "where", "which", "when", "why",
        "who", "how", "tell", "show", "give",
        "find", "explain", "compare", "is", "are",
        "does", "did", "can", "could", "would",
        "the", "and", "for", "about",
        "revenue", "sales", "income", "profit", "profits",
        "net", "operating", "total", "current",
        "expenses", "expenditure", "cost", "costs",
        "cash", "assets", "liabilities", "growth",
        "increase", "decrease", "year", "financial",
        "industry", "sector", "headquarters", "location",
        "total", "annual", "report", "document", "company",
    }

    matches = re.findall(
        r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b",
        query,
    )

    entities = []

    for match in matches:
        if match.lower() in ignored:
            continue

        first_word = match.split()[0].lower()

        if first_word in ignored:
            remaining = " ".join(match.split()[1:])
            if not remaining:
                continue
            match = remaining

        if all(
            word.lower() in ignored
            or re.fullmatch(
                r"(?:fy)?20\d{2}",
                word,
                re.IGNORECASE,
            )
            for word in match.split()
        ):
            continue

        if match not in entities:
            entities.append(match)

    return entities


def is_cross_company_query(query):
    patterns = [
        r"\bwhich company\b",
        r"\bwhich companies\b",
        r"\bcompare\b",
        r"\ball companies\b",
        r"\beach company\b",
        r"\bevery company\b",
        r"\bacross all\b",
        r"\ball (?:uploaded )?(?:files|documents|reports)\b",
        r"\beach (?:uploaded )?(?:file|document|report)\b",
        r"\bevery (?:uploaded )?(?:file|document|report)\b",
        r"\bamong all\b",
        r"\bhighest\b",
        r"\blowest\b",
        r"\bmost profitable\b",
        r"\bleast profitable\b",
        r"\bcommon trends\b",
        r"\bcompany had\b",
        r"\bcompany has\b",
    ]

    return any(
        re.search(pattern, query.lower())
        for pattern in patterns
    )


# --------------------------------------------------
# RELEVANCE SCORING
# --------------------------------------------------

def calculate_relevance(query, content):
    query_lower = query.lower()
    content_lower = content.lower()

    query_words = set(
        re.findall(r"[a-z0-9]+", query_lower)
    )

    stop_words = {
        "which", "company", "companies", "what",
        "was", "were", "the", "a", "an", "is",
        "are", "of", "in", "on", "for", "to",
        "between", "and", "did", "does", "has",
        "have", "all", "their", "its", "how",
        "much", "many", "me", "explain", "had",
        "with", "from", "that", "this", "than",
        "during", "reported", "report", "where",
        "when", "who", "about", "located",
    }

    query_words -= stop_words

    content_words = set(
        re.findall(r"[a-z0-9]+", content_lower)
    )

    score = len(
        query_words.intersection(content_words)
    )

    concepts = {
        "revenue": [
            "revenue", "sales", "total income",
            "operating revenue", "total revenue",
        ],
        "net profit": [
            "net profit", "net income",
            "profit after tax", "profit after taxation",
            "profit for the year",
        ],
        "profit": [
            "net profit", "net income",
            "profit after tax", "operating profit",
            "profit before tax",
        ],
        "expenses": [
            "expenses", "expenditure",
            "operating costs", "total expenses",
        ],
        "cash": [
            "cash balance", "cash reserves",
            "cash equivalents",
        ],
        "assets": [
            "total assets", "assets", "current assets",
        ],
        "liabilities": [
            "liabilities", "total liabilities",
            "current liabilities",
        ],
        "growth": [
            "growth", "increase", "increased",
            "year-on-year", "year over year",
        ],
        "head office": [
            "head office", "headquarters",
            "registered office", "corporate office",
            "located", "location",
        ],
        "headquarters": [
            "headquarters", "head office",
            "registered office", "corporate office",
        ],
        "location": [
            "location", "located", "head office",
            "headquarters", "registered office",
        ],
        "industry": [
            "industry", "sector", "business sector",
        ],
    }

    for concept, alternatives in concepts.items():
        if concept in query_lower:
            if any(term in content_lower for term in alternatives):
                score += 12

    years = re.findall(
        r"\b(?:fy\s*)?(20\d{2})\b",
        query_lower,
    )

    for year in set(years):
        if year in content_lower:
            score += 8

    return score


# --------------------------------------------------
# USER DOCUMENT FETCHING
# --------------------------------------------------

def fetch_user_documents(user_id):
    """
    Fetch only documents owned by the authenticated user.
    This does not modify or rebuild ChromaDB.
    """

    result = db.get(
        where={"user_id": user_id},
        include=["documents", "metadatas"],
    )

    documents = []

    contents = result.get("documents") or []
    metadatas = result.get("metadatas") or []

    for content, metadata in zip(contents, metadatas):
        if not content:
            continue

        documents.append(
            Document(
                page_content=content,
                metadata=metadata or {},
            )
        )

    return documents


# --------------------------------------------------
# COMPANY-SPECIFIC RETRIEVAL
# --------------------------------------------------

def retrieve_company_docs(
    query,
    user_id,
    k=10,
    debug=False,
):
    """
    Return:
      None -> no company was specified; semantic retrieval may be used.
      []   -> a company was specified but not found.
      docs -> documents belonging to the requested company.
    """

    all_documents = fetch_user_documents(user_id)

    if not all_documents:
        return []

    company_names = {}

    for doc in all_documents:
        company = get_company_identifier(doc.metadata)

        if company:
            company_names[normalize(company)] = company

    normalized_query = normalize(query)

    matching_companies = set()

    # Match known company names against the question.
    for normalized_company in company_names:
        if (
            normalized_company
            and normalized_company in normalized_query
        ):
            matching_companies.add(normalized_company)

    # Detect an explicitly named company that is not uploaded.
    named_entities = extract_named_entities(query)

    if not matching_companies and named_entities:
        if debug:
            print(
                "Named entity not found in uploaded company names:",
                named_entities,
            )

        return []

    # No company name in the question.
    if not matching_companies:
        return None

    company_documents = []

    for doc in all_documents:
        company = get_company_identifier(doc.metadata)

        if normalize(company) in matching_companies:
            company_documents.append(doc)

    if not company_documents:
        return []

    ranked_documents = []

    for doc in company_documents:
        relevance = calculate_relevance(
            query,
            doc.page_content,
        )

        ranked_documents.append((doc, relevance))

    ranked_documents.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    relevant_documents = [
        doc
        for doc, score in ranked_documents
        if score > 0
    ]

    # If the metric words are not present in a chunk,
    # still search the other chunks of this same company.
    if not relevant_documents:
        relevant_documents = [
            doc for doc, score in ranked_documents
        ]

    selected_docs = relevant_documents[:k]

    if debug:
        print("\n========== COMPANY RETRIEVAL ==========")
        print("Question:", query)
        print("Matched companies:", matching_companies)
        print("Company chunks:", len(company_documents))
        print("Selected chunks:", len(selected_docs))

        for doc in selected_docs:
            print("Source:", doc.metadata.get("source", "Unknown"))

        print("=======================================\n")

    return selected_docs


# --------------------------------------------------
# CROSS-COMPANY RETRIEVAL
# --------------------------------------------------

def compact_evidence(document, query, max_chars=700):
    """
    Keep relevant lines and nearby lines so that table
    values are not separated from their metric labels.
    """

    lines = [
        line.strip()
        for line in document.page_content.splitlines()
        if line.strip()
    ]

    if not lines:
        return document.page_content[:max_chars]

    query_lower = query.lower()

    terms = {
        term
        for term in [
            "revenue", "sales", "net profit", "net income",
            "profit", "expenses", "assets", "liabilities",
            "cash", "growth", "head office", "headquarters",
            "registered office", "corporate office",
            "location", "located", "address",
        ]
        if term in query_lower
    }

    stop_words = {
        "which", "company", "companies", "what", "where",
        "when", "who", "how", "has", "have", "had", "its",
        "their", "the", "a", "an", "is", "are", "of", "in",
        "on", "for", "to", "and", "or", "with", "was", "were",
    }

    terms.update(
        word
        for word in re.findall(r"[a-z0-9]+", query_lower)
        if word not in stop_words and len(word) > 2
    )

    years = re.findall(
        r"\b(?:fy\s*)?20\d{2}\b",
        query_lower,
    )

    selected_indexes = set()

    for index, line in enumerate(lines):
        lower_line = line.lower()

        if (
            any(term in lower_line for term in terms)
            or any(year in lower_line for year in years)
        ):
            # Include adjacent lines because financial tables
            # often put labels and values on separate lines.
            for nearby in range(
                max(0, index - 1),
                min(len(lines), index + 3),
            ):
                selected_indexes.add(nearby)

    if not selected_indexes:
        return "\n".join(lines)[:max_chars]

    selected = "\n".join(
        lines[index]
        for index in sorted(selected_indexes)
    )

    return selected[:max_chars]


def retrieve_cross_company_docs(
    query,
    user_id,
    debug=False,
):
    all_documents = fetch_user_documents(user_id)

    if not all_documents:
        return []

    source_chunks = defaultdict(list)

    for index, doc in enumerate(all_documents):
        source = doc.metadata.get(
            "source",
            f"unknown_source_{index}",
        )

        relevance = calculate_relevance(
            query,
            doc.page_content,
        )

        if relevance > 0:
            source_chunks[source].append({
                "document": doc,
                "relevance": relevance,
            })

    chunks_per_source = 2

    selected_docs = []

    for source, chunks in source_chunks.items():
        chunks.sort(
            key=lambda item: item["relevance"],
            reverse=True,
        )

        for item in chunks[:chunks_per_source]:
            original_doc = item["document"]

            metadata = original_doc.metadata.copy()
            metadata["retrieval_scope"] = "cross_company"

            compact_doc = Document(
                page_content=compact_evidence(
                    original_doc,
                    query,
                ),
                metadata=metadata,
            )

            selected_docs.append(compact_doc)

    if debug:
        print("\n========== CROSS-COMPANY RETRIEVAL ==========")
        print("Question:", query)
        print("User's stored chunks:", len(all_documents))
        print("Matching source files:", len(source_chunks))
        print("Selected chunks:", len(selected_docs))
        print("=============================================\n")

    return selected_docs


# --------------------------------------------------
# MAIN RETRIEVAL
# --------------------------------------------------

def retrieve_docs(
    query: str,
    user_id: str,
    k=10,
    threshold=0.55,
    debug=False,
):
    if not user_id:
        raise ValueError("A valid user_id is required.")

    # 1. Cross-company questions.
    if is_cross_company_query(query):
        relevant_docs = retrieve_cross_company_docs(
            query=query,
            user_id=user_id,
            debug=debug,
        )

        mode = "CROSS-COMPANY"

    else:
        # 2. Company-specific questions.
        company_docs = retrieve_company_docs(
            query=query,
            user_id=user_id,
            k=k,
            debug=debug,
        )

        if company_docs is not None:
            relevant_docs = company_docs
            mode = "COMPANY-SPECIFIC"

        else:
            # 3. Semantic retrieval only when no company
            # was explicitly named in the question.
            results = db.similarity_search_with_score(
                query=query,
                k=k,
                filter={"user_id": user_id},
            )

            relevant_results = [
                (doc, score)
                for doc, score in results
                if score < threshold
            ]

            # Keep the nearest candidates when the configured cutoff
            # rejects every result; the generator still must ground its
            # answer in the returned evidence.
            if results and not relevant_results:
                relevant_results = results[:min(k, 5)]

            relevant_results.sort(
                key=lambda item: (
                    calculate_relevance(
                        query,
                        item[0].page_content,
                    ),
                    -item[1],
                ),
                reverse=True,
            )

            relevant_docs = [
                doc for doc, score in relevant_results
            ]

            mode = "SEMANTIC"

    if debug:
        print("\n========== FINRAG RETRIEVAL ==========")
        print("Question:", query)
        print("Mode:", mode)
        print("Retrieved chunks:", len(relevant_docs))

        for doc in relevant_docs:
            print("Source:", doc.metadata.get("source", "Unknown"))

        print("======================================\n")

    return relevant_docs
