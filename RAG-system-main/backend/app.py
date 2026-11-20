
import logging
import os
import re
from typing import Optional

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Depends,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.security import (
    authenticate_user,
    create_access_token,
    require_authenticated_user,
    register_user,
)

from retrieval.retriever import retrieve_docs, db
from generation.answer_generation import generate_answer
from ingestion.upload_processor import save_and_process_upload


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NOT_FOUND = "Information not found."


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="FinRAG AI",
    description="Offline Financial Document Intelligence System",
    version="1.0.0",
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


# --------------------------------------------------
# CONVERSATION MEMORY
# --------------------------------------------------

# Each session has its own message history and active entity.
# Memory resets when the backend restarts.

conversation_memory = {}
conversation_context = {}


# --------------------------------------------------
# REQUEST MODELS
# --------------------------------------------------

class RegisterRequest(BaseModel):
    full_name: str = Field(min_length=2, max_length=100)
    username: str = Field(min_length=3, max_length=30)
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=8, max_length=128)
    confirm_password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    username: str
    password: str


class Query(BaseModel):
    question: str
    session_id: Optional[str] = "default"


# --------------------------------------------------
# TEXT NORMALIZATION
# --------------------------------------------------

def normalize(value):
    return re.sub(
        r"[^a-z0-9]",
        "",
        str(value).lower(),
    )


def get_document_entity(metadata):
    """
    Get the company/document identity from metadata or filename.
    """

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


# --------------------------------------------------
# GET USER'S UPLOADED ENTITIES
# --------------------------------------------------

def get_uploaded_entities(user_id):
    """
    Collect unique company/document names from the current
    user's uploaded documents.
    """

    result = db.get(
        where={"user_id": user_id},
        include=["metadatas"],
    )

    entities = {}

    for metadata in result.get("metadatas") or []:
        metadata = metadata or {}

        name = get_document_entity(metadata)

        if name:
            entities[normalize(name)] = name

    return entities


# --------------------------------------------------
# EXPLICIT ENTITY DETECTION
# --------------------------------------------------

def find_explicit_entity(question, uploaded_entities):
    """
    Find an uploaded company/document name explicitly
    mentioned in the question.
    """

    normalized_question = normalize(question)

    matches = []

    for normalized_name, original_name in uploaded_entities.items():
        if (
            normalized_name
            and normalized_name in normalized_question
        ):
            matches.append(original_name)

    if not matches:
        return None

    # Prefer the longest matching name.
    return max(matches, key=len)


def detect_new_entity_name(question):
    """
    Detect a likely new proper name even if it is not
    present in the user's uploaded documents.

    This prevents a missing company from accidentally
    inheriting the previous company's context.
    """

    ignored = {
        "what", "where", "which", "when", "why",
        "who", "how", "tell", "show", "give",
        "find", "explain", "compare", "is", "are",
        "does", "did", "can", "could", "would",
        "the", "and", "for", "about", "according",
        "please", "describe", "summarize",
        "revenue", "sales", "income", "profit", "profits",
        "net", "operating", "total", "current",
        "expenses", "expenditure", "cost", "costs",
        "cash", "assets", "liabilities", "growth",
        "increase", "decrease", "year", "financial",
        "industry", "sector", "headquarters", "location",
        "report", "document", "company", "business",
    }

    matches = re.findall(
        r"\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b",
        question,
    )

    candidates = []

    for match in matches:
        words = match.split()

        words = [
            word for word in words
            if (
                word.lower() not in ignored
                and not re.fullmatch(
                    r"(?:fy)?20\d{2}",
                    word,
                    re.IGNORECASE,
                )
            )
        ]

        candidate = " ".join(words).strip()

        if candidate and candidate not in candidates:
            candidates.append(candidate)

    if not candidates:
        return None

    return max(candidates, key=len)


# --------------------------------------------------
# FOLLOW-UP DETECTION
# --------------------------------------------------

def is_year_followup(question):
    return bool(
        re.match(
            r"^\s*(what about|how about|and|same for|for)\b",
            question,
            re.IGNORECASE,
        )
        and re.search(
            r"\b(?:fy\s*)?20\d{2}\b",
            question,
            re.IGNORECASE,
        )
    )


def is_entity_followup(question):
    patterns = [
        r"\bthat company\b",
        r"\bthis company\b",
        r"\bthe company\b",
        r"\bthat document\b",
        r"\bthis document\b",
        r"\bthat business\b",
        r"\bthis business\b",
        r"\bthat organization\b",
        r"\bthis organization\b",
        r"\bthat one\b",
        r"\bthis one\b",
        r"\bit\b",
        r"\bits\b",
        r"\btheir\b",
        r"\babout it\b",
        r"\bexplain it\b",
    ]

    return any(
        re.search(pattern, question, re.IGNORECASE)
        for pattern in patterns
    )


def is_cross_company_question(question):
    patterns = [
        r"\bcompare\b",
        r"\ball companies\b",
        r"\beach company\b",
        r"\bevery company\b",
        r"\bwhich companies\b",
        r"\bwhich company\b",
        r"\ball (?:uploaded )?(?:files|documents|reports)\b",
        r"\beach (?:uploaded )?(?:file|document|report)\b",
        r"\bevery (?:uploaded )?(?:file|document|report)\b",
        r"\bhighest\b",
        r"\blowest\b",
        r"\bamong all\b",
        r"\bacross all\b",
        r"\bmost profitable\b",
        r"\bleast profitable\b",
        r"\bcompany had\b",
        r"\bcompany has\b",
        r"\bcommon trends\b",
    ]

    return any(
        re.search(pattern, question, re.IGNORECASE)
        for pattern in patterns
    )


# --------------------------------------------------
# FOLLOW-UP QUESTION RESOLUTION
# --------------------------------------------------

def resolve_entity_followup(question, active_entity):
    """
    Replace references to the previously discussed entity.
    """

    if not active_entity:
        return None

    resolved = question

    replacements = [
        (
            r"\bthat company\b|\bthis company\b|\bthe company\b",
            active_entity,
        ),
        (
            r"\bthat document\b|\bthis document\b",
            active_entity,
        ),
        (
            r"\bthat business\b|\bthis business\b",
            active_entity,
        ),
        (
            r"\bthat organization\b|\bthis organization\b",
            active_entity,
        ),
        (
            r"\bthat one\b|\bthis one\b",
            active_entity,
        ),
        (
            r"\babout it\b",
            f"about {active_entity}",
        ),
        (
            r"\bexplain it\b",
            f"explain {active_entity}",
        ),
        (
            r"\bit\b",
            active_entity,
        ),
        (
            r"\bits\b",
            f"{active_entity}'s",
        ),
        (
            r"\btheir\b",
            f"{active_entity}'s",
        ),
    ]

    for pattern, replacement in replacements:
        resolved = re.sub(
            pattern,
            replacement,
            resolved,
            flags=re.IGNORECASE,
        )

    return resolved


def resolve_year_followup(question, last_question):
    """
    Resolve:
    Previous: What was AsterNova's revenue in FY2023?
    Current: What about FY2024?
    """

    if not last_question:
        return None

    year_match = re.search(
        r"\b(?:fy\s*)?(20\d{2})\b",
        question,
        re.IGNORECASE,
    )

    if not year_match:
        return None

    year = year_match.group(1)

    previous_year_pattern = r"\b(?:fy\s*)?20\d{2}\b"

    if re.search(
        previous_year_pattern,
        last_question,
        re.IGNORECASE,
    ):
        return re.sub(
            previous_year_pattern,
            f"FY{year}",
            last_question,
            count=1,
            flags=re.IGNORECASE,
        )

    return (
        f"{last_question.rstrip('?.!')} in FY{year}?"
    )


def resolve_short_followup(question, active_entity, last_question):
    """
    Turn short metric follow-ups into searchable standalone questions.
    """

    prefix = re.match(
        r"^\s*(?:what about|how about|and|same for)\b",
        question,
        re.IGNORECASE,
    )

    if not prefix or not active_entity:
        return None

    followup = question[prefix.end():].strip(" \t?.!")

    if not followup:
        return None

    followup = resolve_entity_followup(
        followup,
        active_entity,
    )

    if normalize(active_entity) not in normalize(followup):
        followup = f"{followup} for {active_entity}"

    if not re.search(
        r"\b(?:fy\s*)?20\d{2}\b",
        followup,
        re.IGNORECASE,
    ):
        previous_year = re.search(
            r"\b(?:fy\s*)?(20\d{2})\b",
            last_question or "",
            re.IGNORECASE,
        )
        if previous_year:
            followup = f"{followup} in FY{previous_year.group(1)}"

    return f"What is {followup}?"


# --------------------------------------------------
# COMPLETE QUESTION RESOLUTION
# --------------------------------------------------

def resolve_user_question(
    question,
    user_id,
    context,
):
    """
    Resolve standalone questions and follow-ups.

    Returns:
        resolved_question
        active_entity
        clarification_message
    """

    uploaded_entities = get_uploaded_entities(user_id)

    active_entity = context.get("active_entity")
    last_question = context.get("last_question")

    # Cross-company questions should not inherit
    # the previous single-company context.
    if is_cross_company_question(question):
        return question, active_entity, None

    # 1. An explicitly mentioned uploaded entity
    # always takes priority over previous conversation.
    explicit_entity = find_explicit_entity(
        question,
        uploaded_entities,
    )

    if explicit_entity:
        context["active_entity"] = explicit_entity

        return question, explicit_entity, None

    # 2. Detect a new entity even if it is not uploaded.
    # Do not fall back to an older company's context.
    new_entity = detect_new_entity_name(question)

    if new_entity:
        context["active_entity"] = new_entity

        return question, new_entity, None

    # 3. Resolve a financial-year follow-up.
    if is_year_followup(question):
        resolved = resolve_year_followup(
            question,
            last_question,
        )

        if resolved:
            return resolved, active_entity, None

    # 3. Resolve short metric follow-ups before treating
    #    their capitalized first word as a new entity.
    resolved = resolve_short_followup(
        question,
        active_entity,
        last_question,
    )

    if resolved:
        return resolved, active_entity, None

    # 4. Resolve pronouns and references.
    if is_entity_followup(question):
        resolved = resolve_entity_followup(
            question,
            active_entity,
        )

        if resolved:
            return resolved, active_entity, None

        return (
            question,
            active_entity,
            (
                "Which company or document are you referring to? "
                "Please mention its name so I can retrieve the "
                "correct information."
            ),
        )

    # Keep unqualified follow-up questions scoped to the current
    # company instead of searching across every uploaded document.
    if active_entity and last_question:
        resolved = question.rstrip("?.! ")
        if normalize(active_entity) not in normalize(resolved):
            resolved = f"{resolved} for {active_entity}"
        return f"{resolved}?", active_entity, None

    # 5. No active company context.
    return question, active_entity, None


# --------------------------------------------------
# AUTOMATIC UPLOAD EVALUATION
# --------------------------------------------------

def evaluate_uploaded_document(
    upload_result: dict,
    user_id: str,
) -> dict:

    filename = upload_result.get("filename", "")
    modality = upload_result.get("modality", "")

    documents_loaded = upload_result.get(
        "documents_loaded", 0
    )

    chunks_added = upload_result.get(
        "chunks_added", 0
    )

    checks = []

    def add_check(name, passed, details):
        checks.append({
            "name": name,
            "status": "passed" if passed else "failed",
            "passed": bool(passed),
            "details": details,
        })

    # Extraction
    extraction_passed = (
        isinstance(documents_loaded, int)
        and documents_loaded > 0
    )

    add_check(
        "Text extraction",
        extraction_passed,
        (
            f"{documents_loaded} document section(s) extracted."
            if extraction_passed
            else "No readable document sections were extracted."
        ),
    )

    # Chunk creation
    chunking_passed = (
        isinstance(chunks_added, int)
        and chunks_added > 0
    )

    add_check(
        "Chunk creation",
        chunking_passed,
        (
            f"{chunks_added} chunk(s) created."
            if chunking_passed
            else "No usable chunks were created."
        ),
    )

    # Metadata and ownership
    metadata_passed = False
    indexed_chunks = 0
    readable_chunks = 0

    try:
        stored = db.get(
            where={"user_id": user_id},
            include=["documents", "metadatas"],
        )

        stored_documents = stored.get("documents", [])
        stored_metadatas = stored.get("metadatas", [])

        for content, metadata in zip(
            stored_documents,
            stored_metadatas,
        ):
            metadata = metadata or {}

            if metadata.get("source") != filename:
                continue

            if metadata.get("user_id") != user_id:
                continue

            indexed_chunks += 1

            if (
                metadata.get("modality") == modality
                and isinstance(content, str)
                and content.strip()
            ):
                readable_chunks += 1

        metadata_passed = (
            indexed_chunks >= chunks_added
            and readable_chunks >= chunks_added
        )

        add_check(
            "Metadata and ownership",
            metadata_passed,
            (
                f"{readable_chunks} readable chunk(s) verified."
                if metadata_passed
                else
                f"Expected {chunks_added} correctly tagged chunks; "
                f"found {readable_chunks}."
            ),
        )

    except Exception:
        logger.exception("Metadata evaluation failed.")

        add_check(
            "Metadata and ownership",
            False,
            "Unable to verify document metadata.",
        )

    # Indexing
    indexing_passed = (
        chunks_added > 0
        and indexed_chunks >= chunks_added
    )

    add_check(
        "ChromaDB indexing",
        indexing_passed,
        (
            f"{indexed_chunks} matching chunks found."
            if indexing_passed
            else "Expected chunks could not be verified."
        ),
    )

    total_checks = len(checks)

    passed_checks = sum(
        1 for check in checks
        if check["passed"]
    )

    score = round(
        (passed_checks / total_checks) * 100,
        2,
    ) if total_checks else 0

    all_passed = passed_checks == total_checks

    return {
        "filename": filename,
        "modality": modality,
        "status": "passed" if all_passed else "needs_attention",
        "score": score,
        "passed_checks": passed_checks,
        "total_checks": total_checks,
        "checks": checks,
        "answer_accuracy": {
            "status": "not_evaluated",
            "score": None,
            "message": (
                "Verified reference answers are required "
                "to calculate genuine answer accuracy."
            ),
        },
    }


# --------------------------------------------------
# HOME AND HEALTH
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "FinRAG AI backend is running",
        "status": "success",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


# --------------------------------------------------
# REGISTRATION
# --------------------------------------------------

@app.post("/register")
def register(credentials: RegisterRequest):

    if credentials.password != credentials.confirm_password:
        raise HTTPException(
            status_code=400,
            detail="Passwords do not match.",
        )

    try:
        user = register_user(
            full_name=credentials.full_name,
            username=credentials.username,
            email=credentials.email,
            password=credentials.password,
        )

        logger.info(
            "New user registered: %s",
            user["username"],
        )

        return {
            "message": (
                "Registration successful. You can now sign in."
            ),
            "user": user,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


# --------------------------------------------------
# LOGIN
# --------------------------------------------------

@app.post("/login")
def login(credentials: LoginRequest):

    user = authenticate_user(
        credentials.username,
        credentials.password,
    )

    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password.",
        )

    token = create_access_token(user)

    logger.info(
        "User logged in: %s",
        user["username"],
    )

    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": 1800,
        "user": {
            "full_name": user["full_name"],
            "username": user["username"],
            "email": user["email"],
            "role": user["role"],
        },
    }


# --------------------------------------------------
# ASK QUESTIONS
# --------------------------------------------------

@app.post("/ask")
def ask(
    query: Query,
    current_user: str = Depends(require_authenticated_user),
):

    question = query.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question.",
        )

    client_session_id = query.session_id or "default"

    session_id = f"{current_user}:{client_session_id}"

    history = conversation_memory.get(session_id, [])

    context = conversation_context.setdefault(
        session_id,
        {
            "active_entity": None,
            "last_question": None,
        },
    )

    try:
        logger.info("Question received.")

        # --------------------------------------------------
        # RESOLVE CURRENT QUESTION
        # --------------------------------------------------

        retrieval_query, active_entity, clarification = (
            resolve_user_question(
                question=question,
                user_id=current_user,
                context=context,
            )
        )

        context["active_entity"] = active_entity

        if clarification:
            answer = clarification

            history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ])

            conversation_memory[session_id] = history[-10:]

            return {
                "question": question,
                "answer": answer,
                "documents_retrieved": 0,
                "session_id": client_session_id,
            }

        logger.info(
            "Resolved question: %s",
            retrieval_query,
        )

        logger.info(
            "Active entity: %s",
            context.get("active_entity"),
        )

        # --------------------------------------------------
        # VERIFY EXPLICIT ENTITY EXISTS IN USER DOCUMENTS
        # --------------------------------------------------

        if active_entity:
            uploaded_entities = get_uploaded_entities(current_user)

            normalized_entity = normalize(active_entity)

            known_entity = any(
                normalized_entity == stored_name
                for stored_name in uploaded_entities
            )

            if not known_entity:
                answer = NOT_FOUND

                history.extend([
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ])

                conversation_memory[session_id] = history[-10:]

                context["last_question"] = retrieval_query

                return {
                    "question": question,
                    "answer": answer,
                    "documents_retrieved": 0,
                    "session_id": client_session_id,
                }

        # --------------------------------------------------
        # RETRIEVE EVIDENCE
        # --------------------------------------------------

        docs = retrieve_docs(
            query=retrieval_query,
            user_id=current_user,
            debug=True,
        )

        logger.info(
            "Retrieved %s relevant chunks.",
            len(docs),
        )

        # --------------------------------------------------
        # GENERATE ANSWER
        # --------------------------------------------------

        if docs:
            # Do not send old conversation answers as evidence.
            answer = generate_answer(
                query=retrieval_query,
                relevant_docs=docs,
                chat_history=None,
            )
        else:
            answer = NOT_FOUND

        if not answer or not str(answer).strip():
            answer = NOT_FOUND

        # --------------------------------------------------
        # UPDATE SESSION CONTEXT
        # --------------------------------------------------

        context["last_question"] = retrieval_query

        history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])

        conversation_memory[session_id] = history[-10:]

        return {
            "question": question,
            "answer": answer,
            "documents_retrieved": len(docs),
            "session_id": client_session_id,
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception("Question answering failed.")

        raise HTTPException(
            status_code=500,
            detail="Unable to process the question.",
        )


# --------------------------------------------------
# UPLOAD DOCUMENTS
# --------------------------------------------------

@app.get("/uploads/count")
def get_uploaded_file_count(
    current_user: str = Depends(
        require_authenticated_user
    ),
):
    result = db.get(
        where={"user_id": current_user},
        include=["metadatas"],
    )

    filenames = {
        str(metadata.get("source")).strip()
        for metadata in result.get("metadatas") or []
        if metadata and metadata.get("source")
    }

    return {"uploaded_files": len(filenames)}


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    current_user: str = Depends(
        require_authenticated_user
    ),
):

    try:
        result = await save_and_process_upload(
            file,
            user_id=current_user,
        )

        logger.info(
            "Document successfully indexed. Starting evaluation."
        )

        try:
            evaluation = evaluate_uploaded_document(
                upload_result=result,
                user_id=current_user,
            )

        except Exception:
            logger.exception("Automatic evaluation failed.")

            evaluation = {
                "filename": result.get("filename"),
                "status": "evaluation_failed",
                "score": None,
                "checks": [],
                "answer_accuracy": {
                    "status": "not_evaluated",
                    "score": None,
                },
                "message": (
                    "The document was uploaded successfully, "
                    "but automatic evaluation could not complete."
                ),
            }

        return {
            "message": (
                "Document uploaded and indexed successfully."
            ),
            **result,
            "evaluation": evaluation,
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception("Document upload failed.")

        raise HTTPException(
            status_code=500,
            detail="Unable to process the uploaded document.",
        )
