#!/usr/bin/env python3
"""
Pan-India Court Cause List PDF Parser (Regex Only)
==================================================
A production-ready deterministic extraction system for parsing
Indian court cause list PDFs into structured JSON.

Architecture:
  Layer 1: Text Extraction          (PyMuPDF)
  Layer 2: Intelligent Segmentation (Regex-based)
  Layer 3: Structured Extraction    (Regex-based)
  Layer 4: Validation & Normalization
  Layer 5: Main CLI Pipeline

Usage:
  python cause_list_parser.py <pdf_path> [--output output.json]
"""

import argparse
import json
import logging
import os
import re
import sys
import textwrap
import time
from collections import Counter
from typing import Any

import fitz  # PyMuPDF
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cause_list_parser")


# =========================================================================
# LAYER 1 — TEXT EXTRACTION
# =========================================================================

def extract_text_blocks(pdf_path: str) -> list[dict]:
    """
    Open a PDF and extract text block-wise while preserving reading order.

    Strategy:
      1. Iterate every page and call page.get_text("blocks") which returns
         layout-aware text rectangles in reading order.
      2. Detect repetitive headers/footers — if a block's text appears on
         ≥ 50 % of pages, it is considered a header/footer and stripped.
      3. Return a flat list of dicts:
         {"page": int, "bbox": (x0, y0, x1, y1), "text": str}

    Handles PDFs of up to 500 pages efficiently.
    """
    log.info("Layer 1 — Opening PDF: %s", pdf_path)
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    log.info("Layer 1 — Total pages: %d", total_pages)

    # --- First pass: collect all blocks and count occurrences of each text --
    raw_blocks: list[dict] = []
    text_counter: Counter = Counter()

    for page_idx in range(total_pages):
        page = doc[page_idx]
        blocks = page.get_text("blocks")  # list of (x0, y0, x1, y1, text, block_no, block_type)
        for b in blocks:
            # block_type 0 = text, 1 = image — skip images
            if b[6] != 0:
                continue
            text = b[4].strip()
            if not text:
                continue
            raw_blocks.append({
                "page": page_idx + 1,
                "bbox": b[:4],
                "text": text,
            })
            text_counter[text] += 1

        if (page_idx + 1) % 100 == 0:
            log.info("Layer 1 — Extracted page %d / %d", page_idx + 1, total_pages)

    doc.close()
    log.info("Layer 1 — Raw text blocks extracted: %d", len(raw_blocks))

    # --- Second pass: remove repetitive headers/footers -------------------
    threshold = max(2, total_pages * 0.5)  # appears on ≥ 50 % of pages
    repetitive_texts = {t for t, c in text_counter.items() if c >= threshold}
    if repetitive_texts:
        log.info(
            "Layer 1 — Detected %d repetitive header/footer patterns; removing them",
            len(repetitive_texts),
        )
    filtered = [b for b in raw_blocks if b["text"] not in repetitive_texts]
    log.info("Layer 1 — Blocks after header/footer removal: %d", len(filtered))
    return filtered


# =========================================================================
# LAYER 2 — INTELLIGENT SEGMENTATION
# =========================================================================

# Regex that matches common Indian case-number formats.
# Covers: WA, WP, WMP, CMP, CONT P, SUB APPL, CMSA, CRL, Crl.O.P,
#         W.P., W.A., CRP, SA, RP, BAIL APPLN, SLP, FAO, MFA, RSA, etc.
CASE_NUMBER_PATTERN = re.compile(
    r"""
    (?:
        (?:W\.?P\.?|W\.?A\.?|WP|WA|WMP|CMP|CONT\.?\s*P|SUB\.?\s*APPL|
           CMSA|CRL\.?\s*(?:O\.?P\.?|A\.?|M\.?C\.?|R\.?C\.?|APPEAL|PETITION|M\.?P\.?)?|
           Crl\.O\.P|CRP|SA|RSA|RP|BAIL\s*APPLN|SLP|FAO|MFA|
           ARB\.?\s*(?:O\.?P\.?|A\.?|CASE)?|
           C\.?S\.?|O\.?S\.?|A\.?S\.?|E\.?P\.?|O\.?P\.?|
           C\.?M\.?A\.?|M\.?P\.?|TC|
           WRIT\s*(?:APPEAL|PETITION)|
           CIVIL\s*(?:APPEAL|SUIT|MISC|REVISION|PETITION)|
           CRIM(?:INAL)?\s*(?:APPEAL|PETITION|MISC|REVISION)|
           REVIEW\s*(?:APPLN|PETITION)|
           CONTEMPT\s*(?:CASE|PETITION)|
           PIL|OA|TA|MA|COMP|
           M\.?A\.?T\.?|L\.?P\.?A\.?|C\.?O\.?|
           MISC\.?\s*(?:CASE|APPLN|PETITION)|
           T\.?P\.?|REF\.?\s*(?:CASE)?|
           I\.?T\.?A\.?|S\.?A\.?|A\.?)
        \s*(?:\(?\s*(?:C|Crl|Civil|Criminal|MD|OS|SS|DB|SB|FB|Com|Tax|Lab|Cus)\s*\)?\s*)?
        [/\s.]*(?:No\.?\s*)?
        \d{1,6}
        \s*/\s*\d{4}
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Detects VS / V/S / Vs. separator between petitioner and respondent
VS_PATTERN = re.compile(r"\b(?:VS\.?|V/S\.?|VERSUS)\b", re.IGNORECASE)

# Pattern for item/serial numbers at the start of a line that indicate a new case entry
# e.g. "8.", "9)", "10.", "123."
ITEM_BOUNDARY = re.compile(r"^\s*(\d{1,4})\s*[.)]\s+", re.MULTILINE)

# A line of dashes or equal signs used as a separator in some cause lists
SEPARATOR_LINE = re.compile(r"^[\-=_]{5,}\s*$")

# Pattern: item number followed directly by a case type WITHOUT dot/paren separator
# e.g. "186 CRL OP/4375/2026", "42 WP 1234/2025"
ITEM_CASE_BOUNDARY = re.compile(
    r"^\s*(\d{1,4})\s+"
    r"(?:W\.?P\.?|W\.?A\.?|WP|WA|WMP|CMP|CRL\.?\s*(?:O\.?P\.?|A\.?|M\.?C\.?|M\.?P\.?)?|"
    r"Crl\.O\.P|CRP|SA|RSA|RP|BAIL\s*APPLN|SLP|FAO|MFA|ARB|"
    r"C\.?S\.?|O\.?S\.?|A\.?S\.?|E\.?P\.?|O\.?P\.?|C\.?M\.?A\.?|M\.?P\.?|TC|"
    r"CONT\.?P|SUB\.?APPL|CMSA|CIVIL\s|CRIM|PIL|OA|TA|MA|COMP|"
    r"M\.?A\.?T\.?|L\.?P\.?A\.?|C\.?O\.?|MISC\.?|T\.?P\.?|I\.?T\.?A\.?)",
    re.IGNORECASE,
)


def segment_case_blocks(text_blocks: list[dict]) -> list[str]:
    """
    Walk through the ordered text blocks and split them into individual
    case blocks.

    Strategy (based on Madras HC / Indian court cause list format):
      - A new case block starts when we see an ITEM NUMBER at the start
        of a line (e.g. "1", "2", "8", "12" alone or followed by text).
      - A CASE_NUMBER_PATTERN on its own line also marks a new block
        (fallback for cause lists that don't use item numbers).
      - DASHES (------) are NOT case boundaries — they separate advocate
        blocks within a single case entry.
      - "AND" followed by another case number = connected case; kept
        within the same block.

    Returns a list of plain-text strings, one per case.
    """
    log.info("Layer 2 — Segmenting %d text blocks into case blocks", len(text_blocks))

    # Merge all block texts into a single stream
    full_text = "\n".join(b["text"] for b in text_blocks)

    # --- Pre-regex multi-line join for split tokens ---
    full_text = re.sub(r'\(Filing\s*\n\s*No\.?\)', '(Filing No.)', full_text, flags=re.IGNORECASE)
    full_text = re.sub(r'\(Criminal\s*\n\s*Laws?\)', '(Criminal Laws)', full_text, flags=re.IGNORECASE)
    full_text = re.sub(r'\(Civil\s*\n\s*Laws?\)', '(Civil Laws)', full_text, flags=re.IGNORECASE)

    lines = full_text.split("\n")

    # Pattern for standalone item number on its own line: "8", "12", "123"
    standalone_item = re.compile(r"^\s*(\d{1,4})\s*$")

    case_blocks: list[str] = []
    current_block: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip empty lines (but keep them in existing blocks)
        if not stripped:
            if current_block:
                current_block.append(line)
            i += 1
            continue

        # Check if this line is a standalone item number (e.g. just "8")
        # Look ahead to see if next non-empty line has a case number or text
        is_new_item = False
        if standalone_item.match(stripped):
            is_new_item = True
        elif ITEM_BOUNDARY.match(stripped):
            # Item number with text after: "8. CRL OP..." or "8. Dispensed..."  
            is_new_item = True
        elif ITEM_CASE_BOUNDARY.match(stripped):
            # Item number + case type without dot/paren: "186 CRL OP/4375/2026"
            is_new_item = True
        elif not current_block and CASE_NUMBER_PATTERN.match(stripped):
            # Case number at start when no block exists yet (first case)
            is_new_item = True

        if is_new_item and current_block:
            # Save previous block and start new one
            block_text = "\n".join(current_block).strip()
            if block_text:
                case_blocks.append(block_text)
            current_block = [line]
        else:
            current_block.append(line)

        i += 1

    # Don't forget the last block
    if current_block:
        block_text = "\n".join(current_block).strip()
        if block_text:
            case_blocks.append(block_text)

    log.info("Layer 2 — Segmented into %d case blocks", len(case_blocks))
    return case_blocks


# =========================================================================
# LAYER 3 — REGEX-ONLY STRUCTURED EXTRACTION
# =========================================================================

# Pattern to extract case type + number + year from a case number string
_CASE_PARTS = re.compile(
    r"((?:W\.?P\.?|W\.?A\.?|WP|WA|WMP|CMP|CONT\.?\s*P|SUB\.?\s*APPL|CMSA|"
    r"CRL\.?\s*(?:O\.?P\.?|A\.?|M\.?C\.?|APPEAL|PETITION|M\.?P\.?)?|Crl\.O\.P|CRP|"
    r"SA|RSA|RP|BAIL\s*APPLN|SLP|FAO|MFA|ARB\.?\s*(?:O\.?P\.?|A\.?|CASE)?|"
    r"C\.?S\.?|O\.?S\.?|A\.?S\.?|E\.?P\.?|O\.?P\.?|C\.?M\.?A\.?|M\.?P\.?|TC|"
    r"WRIT\s*(?:APPEAL|PETITION)|"
    r"CIVIL\s*(?:APPEAL|SUIT|MISC|REVISION|PETITION)|"
    r"CRIM(?:INAL)?\s*(?:APPEAL|PETITION|MISC|REVISION)|REVIEW\s*(?:APPLN|PETITION)|"
    r"CONTEMPT\s*(?:CASE|PETITION)|PIL|OA|TA|MA|COMP|M\.?A\.?T\.?|L\.?P\.?A\.?|C\.?O\.?|"
    r"MISC\.?\s*(?:CASE|APPLN|PETITION)|T\.?P\.?|REF\.?\s*(?:CASE)?|I\.?T\.?A\.?|S\.?A\.?|A\.?)"
    r"(?:\s*\(?\s*(?:C|Crl|Civil|Criminal|MD|OS|SS|DB|SB|FB|Com|Tax|Lab|Cus)\s*\)?)?)"
    r"[/\s]*(?:No\.?\s*)?(\d{1,6})\s*/\s*(\d{4})",
    re.IGNORECASE | re.VERBOSE,
)

# Listing type patterns
_LISTING_TYPE = re.compile(
    r"\b(?:FOR\s+)?(ADMISSION|HEARING|MOTION|ORDERS|MISCELLANEOUS|FINAL\s*HEARING|"
    r"ARGUMENTS|DISPOSAL|APPEARANCE|CONSIDERATION|REPORT|MENTIONED|FRESH|AFTER\s*NOTICE|"
    r"STAY|INJUNCTION|URGENT|RE-?LIST|ADJOURNED|PART\s*HEARD)\b",
    re.IGNORECASE,
)

# Item/serial number at start of block
_ITEM_NUMBER = re.compile(r"^\s*(\d{1,4})\s*[.)]\s*", re.MULTILINE)

# Advocate pattern — covers Indian court naming conventions:
# Standard: Adv., Mr., Mrs., Sri., Smt., Dr.
# Tamil Nadu / South India: SR.A. (Senior Advocate), M/S. (Messrs/firm)
# Role-based: FOR PET, FOR RESP, FOR R1, GP (Govt Pleader), APP, AGP, ADD GP
# Firm names: ... ASSOCIATES, ... & ASSOCIATES
_ADVOCATE = re.compile(
    r"(?:"
    r"(?:Adv\.?|Advocate|Mr\.?|Mrs\.?|Ms\.?|Sri\.?|Smt\.?|Dr\.?)\s+[A-Z][A-Za-z.\s\-,()]+"
    r"|(?:SR\.?\s*A\.?|M/S\.?)\s*[A-Z][A-Za-z.\s\-,()]+"
    r"|[A-Z][A-Za-z.\s\-]+\s+ASSOCIATES"
    r")",
    re.IGNORECASE,
)

# Broader pattern to detect if a line is an advocate/legal-role line
# Used to determine where respondent ends and advocates begin
_ADVOCATE_LINE = re.compile(
    r"(?:"
    r"(?:Adv\.?|Advocate|Mr\.?|Mrs\.?|Ms\.?|Sri\.?|Smt\.?|Dr\.?|Sh\.?)\s+[A-Z]"
    r"|(?:SR\.?\s*A\.?|M/S\.?)\s*[A-Z]"
    r"|[A-Z][A-Za-z.\s\-]+\s+ASSOCIATES"
    r"|\bFOR\s+(?:PET|RESP|R\d|APPLICANT|APPELLANT|COMPLAINANT|PLAINTIFF)"
    r"|\b(?:ADD\.?\s*)?G\.?P\.?\b"
    r"|\bA\.?G\.?P\.?\b"
    r"|\bA\.?P\.?P\.?\b"
    r"|\bA\.?O\.?S\.?\b"
    r"|\bPVT\s+NOTICE"
    r"|\bGOVT\.?\s*(?:PLEADER|ADV|ADVOCATE)"
    r"|\bPUBLIC\s+PROSECUTOR"
    r"|\bSTANDING\s+COUNSEL"
    r"|\bAMICUS\s+CURIAE"
    r"|\bPARTY[\s\-]*IN[\s\-]*PERSON"
    r"|\((?:R\d|P\d|A\d|for\s+R|for\s+P)"
    r")",
    re.IGNORECASE,
)


def extract_structured_data_regex(case_blocks: list[str]) -> dict[str, Any]:
    """
    Pure regex-based structured extraction — NO LLM needed.

    Parses each case block using pattern matching. Handles AND-connected
    cases by splitting the block into sub-entries and creating a case
    record for each case number found.

    Fast: processes thousands of cases in seconds.
    """
    log.info("Layer 3 (REGEX) — Extracting from %d case blocks", len(case_blocks))

    all_cases: list[dict] = []

    # Pattern to split on "AND" that precedes a case number
    and_split = re.compile(r"\nAND\s*\n", re.IGNORECASE)

    for block in case_blocks:
        # --- Item number from start of block ---
        item_match = _ITEM_NUMBER.search(block)
        item_number = item_match.group(1) if item_match else ""

        # Split block into sub-blocks at AND boundaries
        sub_blocks = and_split.split(block)

        for sub_idx, sub_block in enumerate(sub_blocks):
            case_entry: dict[str, str] = {
                "item_number": item_number,
                "case_number": "",
                "case_type": "",
                "year": "",
                "petitioner": "",
                "respondent": "",
                "advocates": "",
            }

            # --- Case number, type, year ---
            case_match = _CASE_PARTS.search(sub_block)
            if case_match:
                case_entry["case_type"] = re.sub(r"\s+", "", case_match.group(1)).upper()
                case_entry["year"] = case_match.group(3)
                case_entry["case_number"] = case_match.group(0).strip()

            # --- Petitioner / Respondent / Advocates (split by VS) ---
            vs_match = VS_PATTERN.search(sub_block)
            if vs_match:
                after_vs = sub_block[vs_match.end():].strip()

                # Petitioner = text between case number and VS
                if case_match:
                    pet_start = case_match.end()
                    petitioner_text = sub_block[pet_start:vs_match.start()].strip()
                else:
                    petitioner_text = sub_block[:vs_match.start()].strip()

                # Clean up petitioner — take meaningful lines
                pet_lines = [
                    ln.strip() for ln in petitioner_text.split("\n")
                    if ln.strip()
                    and not _CASE_PARTS.search(ln)
                    and not _ITEM_NUMBER.match(ln)
                    and not SEPARATOR_LINE.match(ln.strip())
                ]
                case_entry["petitioner"] = " ".join(pet_lines).strip()
                # Remove leading metadata like "(Filing No.)", "(Criminal Laws)"
                case_entry["petitioner"] = re.sub(
                    r"^\((?:Filing\s*No\.?|Criminal\s*Laws?|Civil)\)\s*",
                    "", case_entry["petitioner"], flags=re.IGNORECASE
                ).strip()

                # Split lines after VS into respondent vs advocates
                resp_lines = []
                adv_lines = []
                found_advocate = False
                hit_dash = False
                for ln in after_vs.split("\n"):
                    ln = ln.strip()
                    if not ln:
                        continue
                    if SEPARATOR_LINE.match(ln):
                        # Dash separator = everything after this is advocates
                        hit_dash = True
                        continue
                    if _LISTING_TYPE.search(ln):
                        break
                    if hit_dash:
                        # Lines after a dash separator are advocates
                        found_advocate = True
                    elif not found_advocate and _ADVOCATE_LINE.search(ln):
                        found_advocate = True
                    if found_advocate:
                        adv_lines.append(ln)
                    else:
                        resp_lines.append(ln)

                case_entry["respondent"] = " ".join(resp_lines).strip()
                case_entry["advocates"] = "; ".join(adv_lines) if adv_lines else ""

                # --- Fallback: if no advocates found, try broader patterns ---
                if not case_entry["advocates"]:
                    adv_matches = _ADVOCATE.findall(sub_block)
                    if adv_matches:
                        case_entry["advocates"] = "; ".join(m.strip() for m in adv_matches)
            else:
                # No VS found — try to extract advocates from anywhere
                adv_matches = _ADVOCATE.findall(sub_block)
                if adv_matches:
                    case_entry["advocates"] = "; ".join(m.strip() for m in adv_matches)

            # Only add if we found a case number
            if case_entry["case_number"]:
                all_cases.append(case_entry)

    # Build output with court metadata
    output: dict[str, Any] = {
        "court_name": "",
        "date": "",
        "bench": "",
        "court_number": "",
        "cases": all_cases,
    }
    _infer_court_metadata(output, case_blocks)
    log.info("Layer 3 (REGEX) — Total cases extracted: %d", len(all_cases))
    return output


# Default model — switch to "ollama/llama3" for free local inference
DEFAULT_MODEL = "gemini-2.5-flash"


def extract_structured_data(
    case_blocks: list[str],
    batch_size: int = 20,
    model_id: str = DEFAULT_MODEL,
    use_ollama: bool = False,
) -> dict[str, Any]:
    """
    Use LangExtract to extract structured data from each case block.

    Supports:
      - Cloud models: "gemini-2.5-flash" (needs LANGEXTRACT_API_KEY)
      - Local models: "llama3" etc. via Ollama (free, unlimited, no API key)

    Process:
      1. Combine case blocks into batches (keeps LLM calls reasonable).
      2. For each batch, call lx.extract with the prompt & few-shot examples.
      3. Parse the returned extractions into the target JSON schema.

    Returns a dict matching the final output schema.
    """
    log.info("Layer 3 — Model: %s (ollama=%s) | Extracting from %d case blocks", model_id, use_ollama, len(case_blocks))

    # Verify API key for cloud models (not needed for Ollama)
    api_key = os.getenv("LANGEXTRACT_API_KEY", "")
    if not use_ollama and not api_key:
        log.error(
            "LANGEXTRACT_API_KEY not set. "
            "Please set it in your environment or .env file. "
            "Or use --model llama3 --ollama for free local inference."
        )
        sys.exit(1)

    all_cases: list[dict] = []

    # ---------- Process in batches ----------------------------------------
    total_batches = (len(case_blocks) + batch_size - 1) // batch_size
    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(case_blocks))
        batch = case_blocks[start:end]
        batch_text = "\n---\n".join(batch)

        log.info(
            "Layer 3 — Processing batch %d / %d  (%d blocks)",
            batch_idx + 1,
            total_batches,
            len(batch),
        )

        # Build lx.extract kwargs — differs for Ollama vs cloud models
        extract_kwargs = dict(
            text_or_documents=batch_text,
            prompt_description=_EXTRACTION_PROMPT,
            examples=_FEW_SHOT_EXAMPLES,
            model_id=model_id,
            max_workers=1,
        )
        if use_ollama:
            # Ollama needs: model_url, fence_output=False, no schema constraints
            extract_kwargs["model_url"] = "http://localhost:11434"
            extract_kwargs["fence_output"] = False
            extract_kwargs["use_schema_constraints"] = False
        else:
            extract_kwargs["api_key"] = api_key

        # Retry with exponential backoff for rate-limited cloud keys
        max_retries = 3 if not use_ollama else 0
        result = None
        for attempt in range(max_retries + 1):
            try:
                result = lx.extract(**extract_kwargs)
                break  # Success
            except Exception as exc:
                if "429" in str(exc) and attempt < max_retries:
                    wait = 60 * (2 ** attempt)  # 60s, 120s, 240s
                    log.warning(
                        "Layer 3 — Rate limited on batch %d, retrying in %ds (%d/%d)",
                        batch_idx + 1, wait, attempt + 1, max_retries,
                    )
                    time.sleep(wait)
                else:
                    log.error("Layer 3 — Error on batch %d: %s", batch_idx + 1, exc)
                    result = None
                    break

        if result is None:
            continue

        # ---- Parse extractions from result --------------------------------
        # LangExtract returns an AnnotatedDocument with .extractions
        if hasattr(result, "extractions"):
            extractions = result.extractions
        elif isinstance(result, list):
            extractions = result
        else:
            extractions = []

        for ext in extractions:
            attrs = ext.attributes if hasattr(ext, "attributes") else {}
            case_entry = {
                "item_number": attrs.get("item_number", ""),
                "case_number": attrs.get("case_number", ""),
                "case_type": attrs.get("case_type", ""),
                "year": attrs.get("year", ""),
                "petitioner": attrs.get("petitioner", ""),
                "respondent": attrs.get("respondent", ""),
                "advocates": attrs.get("advocates", ""),
            }
            all_cases.append(case_entry)

        log.info(
            "Layer 3 — Batch %d yielded %d extractions",
            batch_idx + 1,
            len(extractions),
        )

    # ---------- Build the top-level output dict ----------------------------
    output: dict[str, Any] = {
        "court_name": "",
        "date": "",
        "bench": "",
        "court_number": "",
        "cases": all_cases,
    }

    # Try to infer court-level metadata from the first few text blocks
    _infer_court_metadata(output, case_blocks)

    log.info("Layer 3 — Total cases extracted: %d", len(all_cases))
    return output


def _infer_court_metadata(output: dict, case_blocks: list[str]) -> None:
    """
    Attempt to fill court_name, date, bench, and court_number from the
    first case block (which usually contains header information).
    Uses simple regex heuristics — no LLM call needed.
    """
    if not case_blocks:
        return

    header_text = case_blocks[0]

    # Court name — look for "HIGH COURT OF ..." or "IN THE HIGH COURT ..."
    court_match = re.search(
        r"(?:IN\s+THE\s+)?HIGH\s+COURT\s+(?:OF\s+[\w\s,]+?)(?=\n|$)",
        header_text,
        re.IGNORECASE,
    )
    if court_match:
        output["court_name"] = court_match.group(0).strip()

    # Date — look for DD/MM/YYYY, DD-MM-YYYY, or "Dated: ..." patterns
    date_match = re.search(
        r"(?:DATE[D]?\s*[:\-]\s*)?(\d{1,2}[\-/\.]\d{1,2}[\-/\.]\d{2,4})",
        header_text,
        re.IGNORECASE,
    )
    if date_match:
        output["date"] = date_match.group(1).strip()

    # Bench — look for "HON'BLE" / "JUSTICE" lines
    bench_match = re.search(
        r"((?:HON['\u2019]?BLE\s+)?(?:MR\.?\s+|MRS\.?\s+|MS\.?\s+|SMT\.?\s+|DR\.?\s+)?JUSTICE\s+[A-Z\s.\-]+)",
        header_text,
        re.IGNORECASE,
    )
    if bench_match:
        output["bench"] = bench_match.group(0).strip()

    # Court number — look for "COURT NO." / "COURT HALL"
    court_no_match = re.search(
        r"COURT\s*(?:NO\.?|HALL)\s*[:\-]?\s*(\d+)",
        header_text,
        re.IGNORECASE,
    )
    if court_no_match:
        output["court_number"] = court_no_match.group(1).strip()


# =========================================================================
# LAYER 4 — VALIDATION & NORMALIZATION
# =========================================================================

# Simple regex to validate that a case number looks plausible
_VALID_CASE_NUMBER = re.compile(r"[A-Z./\s()]+\s*(?:No\.?\s*)?\d+\s*/\s*\d{4}", re.IGNORECASE)

# Confidence weights for each field  (total = 100)
_CONFIDENCE_WEIGHTS = {
    "case_number": 30,
    "petitioner":  20,
    "respondent":  20,
    "advocates":   15,
    "case_type":    8,
    "year":         7,
}


def _compute_confidence(case: dict) -> float:
    """Compute a 0.0-1.0 confidence score based on field completeness."""
    score = 0
    for field, weight in _CONFIDENCE_WEIGHTS.items():
        val = case.get(field, "")
        if val and val.strip():
            score += weight
    return round(score / 100, 2)


def validate_and_normalize(
    data: dict[str, Any],
    confidence_threshold: float = 0.3,
) -> dict[str, Any]:
    """
    Post-process the extracted data:
      1. Normalize case numbers (collapse whitespace, uppercase, trim).
      2. Deduplicate entries by case_number.
      3. Regex-validate case numbers.
      4. Compute confidence score per case.
      5. Filter out cases below the confidence threshold.
      6. Ensure every field is a JSON-serializable string (missing → "").

    Returns the cleaned data dict.
    """
    log.info("Layer 4 — Validating & normalizing %d cases (threshold=%.1f%%)",
             len(data.get("cases", [])), confidence_threshold * 100)

    seen: set[str] = set()
    clean_cases: list[dict] = []

    for case in data.get("cases", []):
        # --- Normalize each string field ----------------------------------
        for key in case:
            if not isinstance(case[key], str):
                case[key] = str(case[key]) if case[key] is not None else ""
            case[key] = case[key].strip()

        # --- Normalize case_number ----------------------------------------
        cn = case.get("case_number", "")
        cn = re.sub(r"\s+", " ", cn).strip().upper()
        case["case_number"] = cn

        # --- Uppercase case_type ------------------------------------------
        case["case_type"] = case.get("case_type", "").upper()

        # --- Deduplicate by case_number -----------------------------------
        if cn and cn in seen:
            log.debug("Layer 4 — Duplicate removed: %s", cn)
            continue
        if cn:
            seen.add(cn)

        # --- Validate case_number with regex (skip invalid) ---------------
        if cn and not _VALID_CASE_NUMBER.search(cn):
            log.warning("Layer 4 — Possibly invalid case number kept: %s", cn)

        # --- Ensure all expected keys exist with string values ------------
        for field in (
            "item_number", "case_number", "case_type", "year",
            "petitioner", "respondent", "advocates",
        ):
            case.setdefault(field, "")

        # --- Compute confidence -------------------------------------------
        conf = _compute_confidence(case)
        case["confidence"] = conf

        if conf < confidence_threshold:
            log.debug("Layer 4 — Low confidence (%.0f%%) skipped: %s", conf * 100, cn)
            continue

        clean_cases.append(case)

    data["cases"] = clean_cases

    # Ensure top-level fields are strings
    for key in ("court_name", "date", "bench", "court_number"):
        data.setdefault(key, "")
        if not isinstance(data[key], str):
            data[key] = str(data[key])

    # Log confidence distribution
    if clean_cases:
        confs = [c["confidence"] for c in clean_cases]
        avg = sum(confs) / len(confs)
        low = min(confs)
        high = max(confs)
        log.info("Layer 4 — Confidence: avg=%.0f%%, min=%.0f%%, max=%.0f%%, kept=%d (threshold=%.0f%%)",
                 avg * 100, low * 100, high * 100, len(clean_cases), confidence_threshold * 100)
    else:
        log.warning("Layer 4 — No cases survived after threshold filter (%.0f%%)", confidence_threshold * 100)

    log.info("Layer 4 — Cases after normalization: %d", len(clean_cases))
    return data


# =========================================================================
# LAYER 5 — MAIN PIPELINE
# =========================================================================

def run_pipeline(
    pdf_path: str,
    confidence_threshold: float = 0.3,
) -> dict[str, Any]:
    """
    Orchestrate the full extraction pipeline:

        PDF  →  extract_text_blocks()
             →  segment_case_blocks()
             →  extract_structured_data_regex()
             →  validate_and_normalize()
             →  return final JSON dict
    """
    log.info("=" * 60)
    log.info("Starting Cause List Extraction Pipeline (REGEX MODE)")
    log.info("PDF: %s", pdf_path)
    log.info("=" * 60)

    # Layer 1 — extract raw text blocks from PDF
    text_blocks = extract_text_blocks(pdf_path)
    if not text_blocks:
        log.error("No text blocks extracted. Is this a scanned (image-based) PDF?")
        return {"court_name": "", "date": "", "bench": "", "court_number": "", "cases": []}

    # Layer 2 — segment into individual case blocks
    case_blocks = segment_case_blocks(text_blocks)
    if not case_blocks:
        log.warning("No case blocks identified after segmentation.")
        return {"court_name": "", "date": "", "bench": "", "court_number": "", "cases": []}

    # Layer 3 — extract structured data (v3: exclusively Regex)
    structured = extract_structured_data_regex(case_blocks)

    # Layer 4 — validate and normalize
    final = validate_and_normalize(structured, confidence_threshold=confidence_threshold)

    log.info("=" * 60)
    log.info("Pipeline complete — %d cases extracted", len(final.get("cases", [])))
    log.info("=" * 60)
    return final


def main() -> None:
    """CLI entry-point."""
    parser = argparse.ArgumentParser(
        description="Parse Indian court cause list PDFs into structured JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python cause_list_parser.py cause_list.pdf
              python cause_list_parser.py cause_list.pdf --output result.json
              python cause_list_parser.py cause_list.pdf --batch-size 10
        """),
    )
    parser.add_argument(
        "pdf_path",
        help="Path to the cause list PDF file.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Write JSON output to this file (default: print to stdout).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Minimum confidence score 0.0-1.0 to keep a case (default: 0.3 = 30%%).",
    )
    args = parser.parse_args()

    # Validate input file
    if not os.path.isfile(args.pdf_path):
        log.error("File not found: %s", args.pdf_path)
        sys.exit(1)

    # Run the pipeline (exclusively Regex)
    result = run_pipeline(
        args.pdf_path,
        confidence_threshold=args.confidence,
    )

    # Output
    json_str = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_str)
        log.info("JSON written to %s", args.output)
    else:
        print(json_str)


if __name__ == "__main__":
    main()
