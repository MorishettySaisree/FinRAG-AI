
import os
import re
import tempfile
from pathlib import Path

from fastapi import UploadFile, HTTPException

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from PIL import Image
import pytesseract
import whisper

from retrieval.retriever import db


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MAX_FILE_SIZE = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000

ALLOWED_EXTENSIONS = {
    ".txt", ".pdf", ".docx",
    ".png", ".jpg", ".jpeg", ".bmp",
    ".tif", ".tiff", ".webp",
    ".mp3", ".wav", ".m4a", ".flac", ".ogg",
}

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp",
    ".tif", ".tiff", ".webp",
}

AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg",
}

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

FFMPEG_BIN = r"C:\ffmpeg-8.1.1-essentials_build\bin"

if os.path.isdir(FFMPEG_BIN):
    os.environ["PATH"] = (
        FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")
    )


# --------------------------------------------------
# FINANCIAL KEYWORDS
# --------------------------------------------------

FINANCIAL_KEYWORDS = {
    "revenue", "profit", "loss", "income", "expense",
    "expenses", "financial", "finance", "accounting",
    "account", "accounts", "balance sheet", "cash flow",
    "cashflow", "assets", "liabilities", "equity", "debit",
    "credit", "invoice", "invoices", "payment", "payments",
    "transaction", "transactions", "bank statement",
    "interest rate", "tax", "taxation", "gst", "budget",
    "budgets", "investment", "investments", "dividend",
    "dividends", "loan", "loans", "audit", "auditing",
    "accounts payable", "accounts receivable", "balance",
    "turnover", "expenditure", "fiscal year", "financial year",
    "annual report", "quarterly report", "gross margin",
    "net margin", "operating cost", "operating costs",
    "payroll", "portfolio", "valuation", "depreciation",
    "amortization", "financial statement", "financial statements",
}


# --------------------------------------------------
# FINANCIAL CONTENT VALIDATION
# --------------------------------------------------

def validate_financial_content(text: str) -> list:
    """Validate financial content and return matched keywords."""

    normalized_text = re.sub(
        r"\s+", " ", text.lower()
    ).strip()

    if len(normalized_text) < 30:
        raise ValueError(
            "The file contains too little readable text "
            "to verify that it is financial."
        )

    matched_keywords = {
        keyword
        for keyword in FINANCIAL_KEYWORDS
        if re.search(
            r"(?<!\w)" + re.escape(keyword) + r"(?!\w)",
            normalized_text,
        )
    }

    if len(matched_keywords) < 2:
        raise ValueError(
            "Upload rejected. The file does not contain enough "
            "financial-related content. Please upload a financial "
            "report, statement, invoice, or another supported "
            "financial document."
        )

    return sorted(matched_keywords)


# --------------------------------------------------
# FILE SIGNATURE VALIDATION
# --------------------------------------------------

def validate_file_signature(
    file_path: str,
    extension: str,
) -> None:

    with open(file_path, "rb") as file:
        header = file.read(16)

    if not header:
        raise ValueError("The uploaded file is empty.")

    if extension == ".txt":
        return

    valid = True

    if extension == ".pdf":
        valid = header.startswith(b"%PDF-")

    elif extension == ".docx":
        valid = header.startswith(b"PK\x03\x04")

    elif extension == ".png":
        valid = header.startswith(b"\x89PNG\r\n\x1a\n")

    elif extension in {".jpg", ".jpeg"}:
        valid = header.startswith(b"\xff\xd8\xff")

    elif extension == ".bmp":
        valid = header.startswith(b"BM")

    elif extension in {".tif", ".tiff"}:
        valid = (
            header.startswith(b"II*\x00")
            or header.startswith(b"MM\x00*")
            or header.startswith(b"II+\x00")
            or header.startswith(b"MM\x00+")
        )

    elif extension == ".webp":
        with open(file_path, "rb") as file:
            initial = file.read(12)

        valid = (
            len(initial) >= 12
            and initial[:4] == b"RIFF"
            and initial[8:12] == b"WEBP"
        )

    elif extension == ".wav":
        with open(file_path, "rb") as file:
            initial = file.read(12)

        valid = (
            len(initial) >= 12
            and initial[:4] == b"RIFF"
            and initial[8:12] == b"WAVE"
        )

    elif extension == ".flac":
        valid = header.startswith(b"fLaC")

    elif extension == ".ogg":
        valid = header.startswith(b"OggS")

    elif extension == ".m4a":
        valid = len(header) >= 8 and header[4:8] == b"ftyp"

    elif extension == ".mp3":
        valid = (
            header.startswith(b"ID3")
            or (
                len(header) >= 2
                and header[0] == 0xFF
                and (header[1] & 0xE0) == 0xE0
            )
        )

    if not valid:
        raise ValueError(
            "The file content does not match its extension. "
            "Upload the original, correctly formatted file."
        )


# --------------------------------------------------
# WHISPER MODEL
# --------------------------------------------------

_whisper_model = None


def get_whisper_model():

    global _whisper_model

    if _whisper_model is None:
        _whisper_model = whisper.load_model("base")

    return _whisper_model


# --------------------------------------------------
# AUTOMATIC UPLOAD EVALUATION
# --------------------------------------------------

def create_upload_evaluation(
    documents,
    combined_text,
    matched_keywords,
    chunks,
    filename,
    modality,
    user_id,
    indexed_count,
):
    """
    Evaluate extraction, financial validation,
    chunking, metadata and ChromaDB indexing.
    """

    checks = {}

    # CHECK 1: EXTRACTION

    extracted_characters = len(combined_text.strip())

    extracted_documents = sum(
        1
        for document in documents
        if document.page_content.strip()
    )

    extraction_passed = (
        extracted_characters >= 30
        and extracted_documents > 0
    )

    checks["extraction"] = {
        "passed": extraction_passed,
        "documents_extracted": extracted_documents,
        "characters_extracted": extracted_characters,
        "message": (
            "Readable text extracted successfully."
            if extraction_passed
            else "Insufficient readable text extracted."
        ),
    }

    # CHECK 2: FINANCIAL VALIDATION

    financial_passed = len(matched_keywords) >= 2

    checks["financial_validation"] = {
        "passed": financial_passed,
        "matched_keyword_count": len(matched_keywords),
        "matched_keywords": matched_keywords,
        "message": (
            "Financial content validated."
            if financial_passed
            else "Insufficient financial indicators."
        ),
    }

    # CHECK 3: CHUNKING

    chunk_count = len(chunks)

    chunk_passed = (
        chunk_count > 0
        and all(
            chunk.page_content.strip()
            for chunk in chunks
        )
    )

    checks["chunking"] = {
        "passed": chunk_passed,
        "chunks_created": chunk_count,
        "message": (
            "Document chunks created successfully."
            if chunk_passed
            else "Chunking failed or produced empty chunks."
        ),
    }

    # CHECK 4: METADATA

    metadata_passed = bool(chunks) and all(
        chunk.metadata.get("source") == filename
        and chunk.metadata.get("modality") == modality
        and chunk.metadata.get("user_id") == user_id
        for chunk in chunks
    )

    checks["metadata"] = {
        "passed": metadata_passed,
        "message": (
            "Source, modality and user ownership verified."
            if metadata_passed
            else "One or more chunks have incorrect metadata."
        ),
    }

    # CHECK 5: CHROMADB INDEXING

    indexing_passed = (
        chunk_count > 0
        and indexed_count == chunk_count
    )

    checks["indexing"] = {
        "passed": indexing_passed,
        "chunks_expected": chunk_count,
        "chunks_verified": indexed_count,
        "message": (
            "All uploaded chunks verified in ChromaDB."
            if indexing_passed
            else "Some indexed chunks could not be verified."
        ),
    }

    # OVERALL SCORE

    passed_checks = sum(
        1
        for check in checks.values()
        if check["passed"]
    )

    total_checks = len(checks)

    score = round(
        (passed_checks / total_checks) * 100,
        2
    )

    return {
        "status": (
            "passed"
            if passed_checks == total_checks
            else "failed"
        ),
        "score": score,
        "passed_checks": passed_checks,
        "total_checks": total_checks,
        "checks": checks,
        "answer_accuracy": {
            "status": "not_evaluated",
            "message": (
                "Answer accuracy will be evaluated separately "
                "using verified reference answers."
            ),
        },
    }


# --------------------------------------------------
# PROCESS UPLOADED FILE
# --------------------------------------------------

def process_uploaded_file(
    file_path: str,
    filename: str,
    user_id: str,
) -> dict:

    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("A valid user ID is required.")

    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported file type.")

    validate_file_signature(file_path, extension)

    documents = []
    modality = ""

    # TEXT

    if extension == ".txt":

        loader = TextLoader(
            file_path,
            encoding="utf-8",
            autodetect_encoding=True,
        )

        documents = loader.load()
        modality = "text"

    # PDF

    elif extension == ".pdf":

        loader = PyPDFLoader(file_path)

        documents = loader.load()
        modality = "pdf"

    # DOCX

    elif extension == ".docx":

        loader = Docx2txtLoader(file_path)

        documents = loader.load()
        modality = "docx"

    # IMAGE / OCR

    elif extension in IMAGE_EXTENSIONS:

        Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

        try:

            with Image.open(file_path) as image:
                image.verify()

            with Image.open(file_path) as image:
                image = image.convert("RGB")

                extracted_text = pytesseract.image_to_string(
                    image
                )

        except Exception as exc:

            raise ValueError(
                "The image could not be validated or processed."
            ) from exc

        if extracted_text.strip():

            documents = [
                Document(page_content=extracted_text)
            ]

        modality = "image"

    # AUDIO / WHISPER

    elif extension in AUDIO_EXTENSIONS:

        try:

            model = get_whisper_model()

            transcription = model.transcribe(
                file_path,
                fp16=False,
            )

            extracted_text = transcription.get(
                "text", ""
            ).strip()

        except Exception as exc:

            raise ValueError(
                "The audio file could not be processed."
            ) from exc

        if extracted_text:

            documents = [
                Document(page_content=extracted_text)
            ]

        modality = "audio"

    # EXTRACTION VALIDATION

    if not documents:

        raise ValueError(
            "No readable text was found in the uploaded file."
        )

    combined_text = "\n".join(
        document.page_content
        for document in documents
        if document.page_content
    )

    # FINANCIAL VALIDATION

    matched_keywords = validate_financial_content(
        combined_text
    )

    # ADD METADATA

    for document in documents:

        document.metadata["source"] = filename
        document.metadata["modality"] = modality
        document.metadata["user_id"] = user_id

    # CHUNKING

    if modality == "pdf":

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
        )

    else:

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0,
        )

    chunks = splitter.split_documents(documents)

    if not chunks:

        raise ValueError(
            "No usable text chunks were created from this file."
        )

    # VERIFY METADATA BEFORE INDEXING

    for chunk in chunks:

        if (
            chunk.metadata.get("source") != filename
            or chunk.metadata.get("modality") != modality
            or chunk.metadata.get("user_id") != user_id
        ):

            raise ValueError(
                "Document metadata validation failed."
            )

    # ADD TO EXISTING CHROMADB

    # Do not delete, reset or recreate existing data.
    chunk_ids = db.add_documents(chunks)

    # VERIFY NEWLY INDEXED CHUNKS

    indexed_count = 0

    if chunk_ids:

        indexed_records = db.get(
            ids=chunk_ids,
            include=["metadatas"],
        )

        indexed_ids = indexed_records.get("ids", [])

        indexed_metadatas = indexed_records.get(
            "metadatas", []
        )

        # Count only records whose ownership and metadata
        # match this upload.
        for metadata in indexed_metadatas:

            metadata = metadata or {}

            if (
                metadata.get("source") == filename
                and metadata.get("modality") == modality
                and metadata.get("user_id") == user_id
            ):
                indexed_count += 1

    # CREATE EVALUATION REPORT

    evaluation = create_upload_evaluation(
        documents=documents,
        combined_text=combined_text,
        matched_keywords=matched_keywords,
        chunks=chunks,
        filename=filename,
        modality=modality,
        user_id=user_id,
        indexed_count=indexed_count,
    )

    # RETURN UPLOAD + EVALUATION RESULTS

    return {
        "filename": filename,
        "modality": modality,
        "documents_loaded": len(documents),
        "chunks_added": len(chunks),
        "extracted_characters": len(combined_text.strip()),
        "financial_keywords_found": matched_keywords,
        "evaluation": evaluation,
    }


# --------------------------------------------------
# UPLOAD HANDLER
# --------------------------------------------------

async def save_and_process_upload(
    file: UploadFile,
    user_id: str,
) -> dict:

    original_name = file.filename or ""

    filename = Path(
        original_name.replace("\\", "/")
    ).name

    if (
        not filename
        or filename in {".", ".."}
        or "\x00" in filename
    ):

        await file.close()

        raise HTTPException(
            status_code=400,
            detail="Invalid filename.",
        )

    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:

        await file.close()

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. Supported types: "
                "TXT, PDF, DOCX, PNG, JPG, JPEG, BMP, TIF, TIFF, "
                "WEBP, MP3, WAV, M4A, FLAC, OGG."
            ),
        )

    temp_path = None
    total_size = 0

    try:

        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=extension,
            delete=False,
        ) as temp_file:

            temp_path = temp_file.name

            while True:

                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_FILE_SIZE:

                    raise HTTPException(
                        status_code=413,
                        detail="File exceeds the 20 MB upload limit.",
                    )

                temp_file.write(chunk)

        if total_size == 0:

            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        # PROCESS USING AUTHENTICATED USER ID

        result = process_uploaded_file(
            file_path=temp_path,
            filename=filename,
            user_id=user_id,
        )

        return result

    except HTTPException:
        raise

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        print(
            "Upload processing failed:",
            type(exc).__name__,
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "The file could not be processed. Check that it is "
                "valid, readable, and in a supported format."
            ),
        ) from exc

    finally:

        if temp_path and os.path.exists(temp_path):

            try:
                os.remove(temp_path)

            except OSError:

                print(
                    "Warning: Could not remove a temporary upload file."
                )

        await file.close()
