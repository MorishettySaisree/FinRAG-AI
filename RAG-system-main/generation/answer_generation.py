
import re

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage


# --------------------------------------------------
# LOCAL OLLAMA MODEL
# --------------------------------------------------

model = ChatOllama(
    model="llama3",
    temperature=0,
    num_predict=350,
)


NOT_FOUND = "Information not found."


# --------------------------------------------------
# COMPANY NAME DETECTION
# --------------------------------------------------

def extract_named_entities(query):
    ignored = {
        "what", "where", "which", "when", "why",
        "who", "how", "tell", "show", "give",
        "find", "explain", "compare", "is", "are",
        "does", "did", "can", "could", "would",
        "the", "and", "for", "about", "now",
        "all", "each", "every", "uploaded", "files",
        "file", "documents", "document", "reports", "report",
        "companies", "company", "list", "show", "give",
        "matching", "matches", "answers", "across",
        "summarize", "describe", "explain",
        "ceo", "cfo", "cto", "coo", "chairman",
        "chairperson", "president", "director", "founder",
        "owner", "head", "officer", "chief", "executive",
    }

    matches = re.findall(
        r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b",
        query,
    )

    entities = []

    for match in matches:
        words = match.split()

        while words and words[0].lower() in ignored:
            words.pop(0)

        if not words:
            continue

        match = " ".join(words)

        if match not in entities:
            entities.append(match)

    return entities


def normalize(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


# --------------------------------------------------
# CONVERSATION HISTORY
# --------------------------------------------------

def format_history(chat_history):
    if not chat_history:
        return "No previous conversation."

    recent_messages = chat_history[-6:]

    return "\n".join(
        f"{item.get('role', 'unknown')}: "
        f"{item.get('content', '')}"
        for item in recent_messages
    )


def get_previous_standalone_question(chat_history):
    if not chat_history:
        return None

    metric_pattern = re.compile(
        r"\b(revenue|sales|net profit|net income|profit|"
        r"expenses|assets|liabilities|cash|growth|"
        r"turnover|earnings|head office|headquarters|"
        r"location|address|industry|sector)\b",
        re.IGNORECASE,
    )

    for item in reversed(chat_history):
        role = str(item.get("role", "")).lower()
        content = str(item.get("content", "")).strip()

        if role not in ("user", "human"):
            continue

        if metric_pattern.search(content):
            return content

    return None


# --------------------------------------------------
# FOLLOW-UP QUESTION DETECTION
# --------------------------------------------------

def is_followup_question(query):
    patterns = [
        r"^\s*what about\b",
        r"^\s*how about\b",
        r"^\s*and\b",
        r"^\s*what if\b",
        r"^\s*same for\b",
        r"^\s*for fy\s?20\d{2}\b",
        r"^\s*for\s+20\d{2}\b",
        r"^\s*that year\b",
        r"^\s*what about its\b",
        r"^\s*what about their\b",
        r"^\s*what is its\b",
        r"^\s*what is their\b",
        r"^\s*and its\b",
        r"^\s*and their\b",
    ]

    return any(
        re.search(pattern, query, re.IGNORECASE)
        for pattern in patterns
    )


# --------------------------------------------------
# DETERMINISTIC FOLLOW-UP RESOLUTION
# --------------------------------------------------

def resolve_followup_question(query, chat_history):
    if not is_followup_question(query):
        return None

    previous_question = get_previous_standalone_question(
        chat_history
    )

    if not previous_question:
        return None

    year_match = re.search(
        r"\b(?:fy\s*)?(20\d{2})\b",
        query,
        re.IGNORECASE,
    )

    if not year_match:
        return None

    requested_year = year_match.group(1)

    previous_year_pattern = r"\b(?:fy\s*)?20\d{2}\b"

    if re.search(
        previous_year_pattern,
        previous_question,
        re.IGNORECASE,
    ):
        resolved_question = re.sub(
            previous_year_pattern,
            f"FY{requested_year}",
            previous_question,
            count=1,
            flags=re.IGNORECASE,
        )
    else:
        resolved_question = (
            f"{previous_question.rstrip('?.!')} "
            f"in FY{requested_year}?"
        )

    print("\n========== FOLLOW-UP RESOLUTION ==========")
    print("Previous question:", previous_question)
    print("Current question:", query)
    print("Resolved question:", resolved_question)
    print("==========================================\n")

    return resolved_question


# --------------------------------------------------
# QUESTION REWRITING
# --------------------------------------------------

def rewrite_question(query, chat_history=None):
    query = query.strip()

    if not query or not chat_history:
        return query

    # IMPORTANT:
    # An explicitly named company means this is a
    # standalone question. Never replace it using history.
    named_entities = extract_named_entities(query)

    if named_entities:
        print(
            "Standalone company question detected:",
            named_entities,
        )
        return query

    # Resolve simple financial-year follow-ups without
    # asking the LLM to guess the company or metric.
    resolved_question = resolve_followup_question(
        query,
        chat_history,
    )

    if resolved_question:
        return resolved_question

    # Only use LLM rewriting for questions that genuinely
    # depend on previous conversation.
    history_text = format_history(chat_history)

    prompt = f"""
Rewrite the latest question as a standalone search query.

CONVERSATION:
{history_text}

LATEST QUESTION:
{query}

RULES:

1. Use previous messages only to resolve references
   such as "its", "their", or "that company".
2. Preserve the exact company name.
3. Preserve the requested metric.
4. Preserve the financial year.
5. Do not introduce a different company.
6. Do not invent missing details.
7. If the latest question is already independent,
   return it unchanged.
8. Do not answer the question.

Return only the rewritten question.
"""

    try:
        result = model.invoke([
            SystemMessage(
                content=(
                    "You rewrite questions accurately. "
                    "Never replace an explicitly named company "
                    "with a different company."
                )
            ),
            HumanMessage(content=prompt),
        ])

        rewritten = str(result.content).strip()
        rewritten = rewritten.strip('"').strip("'")

        return rewritten if rewritten else query

    except Exception as error:
        print("Question rewriting failed:", error)
        return query


# --------------------------------------------------
# EVIDENCE PREPARATION
# --------------------------------------------------

def prepare_context(relevant_docs):
    context_parts = []
    seen = set()

    for doc in relevant_docs:
        content = doc.page_content.strip()

        if not content or content in seen:
            continue

        seen.add(content)

        source = doc.metadata.get("source", "Unknown source")

        # Keep enough of the chunk to preserve financial tables.
        context_parts.append(
            f"Source: {source}\n"
            f"Company: {doc.metadata.get('company_name', '')}\n"
            f"{content[:2500]}"
        )

    return "\n\n".join(context_parts)[:24000]


# --------------------------------------------------
# ANSWER GENERATION
# --------------------------------------------------

def generate_answer(query, relevant_docs, chat_history=None):

    if not relevant_docs:
        return NOT_FOUND

    context = prepare_context(relevant_docs)

    if not context.strip():
        return NOT_FOUND

    # Verify that an explicitly named company is present
    # in the retrieved evidence before asking the LLM.
    named_entities = extract_named_entities(query)
    normalized_context = normalize(context)

    for entity in named_entities:
        if normalize(entity) not in normalized_context:
            print(
                "Company evidence mismatch:",
                entity,
            )
            return NOT_FOUND

    multiple_sources = any(
        doc.metadata.get("retrieval_scope") == "cross_company"
        for doc in relevant_docs
    )

    multi_answer_rules = (
        """
13. This is a request across multiple uploaded documents.
    Return one separate bullet for every company/document with
    relevant evidence. Use the company name from the evidence or
    source filename, and do not omit a matching source merely
    because another source has a similar answer. Write each bullet
    as a complete sentence that directly answers the question.
"""
        if multiple_sources
        else
        """
13. Include all distinct answers supported by the supplied
    evidence. If multiple relevant facts are present, list them
    separately instead of choosing only one.
"""
    )

    prompt = f"""
You are FinRAG AI, an offline financial document assistant.

Answer the user's question using ONLY the document evidence.

DOCUMENT EVIDENCE:
{context}

USER QUESTION:
{query}

STRICT RULES:

1. Use only the supplied document evidence.
2. Never use previous conversation answers as evidence.
3. Never invent financial values, company details or locations.
4. Answer for the exact company named in the question.
5. Do not substitute another company.
6. Answer for the exact financial year requested.
7. Never mix information from different companies.
8. For comparisons, use only values supported by the evidence.
9. Calculate differences carefully.
10. Preserve the currency and units in the documents.
11. Do not mention document numbers, chunk numbers,
    retrieval counts or internal source labels.
12. If the requested information is missing, incomplete,
    contradictory or cannot be verified, respond exactly:

Information not found.
{multi_answer_rules}

ANSWER STYLE:

- Simple question: 1-3 sentences.
- Multiple supported matches: use a concise bullet list of complete
  sentences, naming each company and its supported answer.
- Company overview: concise paragraph.
- Financial comparison: explain the values and differences.
- Detailed question: use a clear structured explanation.

Every factual statement must be supported by the evidence.

Return only the answer.
"""

    try:
        result = model.invoke([
            SystemMessage(
                content=(
                    "You are FinRAG AI. Answer only from "
                    "the supplied evidence. Never fabricate "
                    "information or substitute a company."
                )
            ),
            HumanMessage(content=prompt),
        ])

        answer = str(result.content).strip()

        if not answer:
            return NOT_FOUND

        if NOT_FOUND.lower() in answer.lower():
            return NOT_FOUND

        return answer

    except Exception as error:
        print("Answer generation failed:", error)
        return "Unable to generate an answer. Please try again."
