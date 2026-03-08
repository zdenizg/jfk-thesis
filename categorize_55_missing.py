import io
import os
import re
import json
import base64
import traceback
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pdf2image import convert_from_path, pdfinfo_from_path
from groq import Groq
from google.cloud import vision
from google.oauth2 import service_account

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
PDF_DIR = BASE_DIR / "jfk_pdfs"
MISSING_FILE = BASE_DIR / "missing_file_ids.txt"
GOOGLE_CREDS_PATH = BASE_DIR / "google_credentials.json"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = RESULTS_DIR / "jfk_categorization_55missing.csv"
CHECKPOINT_FILE = RESULTS_DIR / "checkpoint_55missing.json"

# ---------------------------------------------------------------------------
# Load API keys
# ---------------------------------------------------------------------------
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

_gcp_credentials = service_account.Credentials.from_service_account_file(
    str(GOOGLE_CREDS_PATH),
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)
vision_client = vision.ImageAnnotatorClient(credentials=_gcp_credentials)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
AUTOSAVE_EVERY = 50

CSV_COLUMNS = [
    "file_id", "number_of_pages", "page_number", "filename",
    "document_type", "ocr_difficulty", "includes_handwriting",
    "has_shadowy_background", "document_quality", "text_density",
    "has_stamps", "has_redactions", "has_forms", "has_tables",
    "is_typewritten", "paper_condition", "primary_characteristics", "content",
]

GROQ_PROMPT = (
    "<role>You are a document analysis expert trained to extract structural, "
    "visual, and categorical insights from scanned or photographed documents.</role>\n"
    "<task>Analyze the provided document image and return a structured JSON object.</task>\n"
    "<instructions>\n"
    "* Visually inspect for handwriting, shadows, stamps, redactions, forms, tables, paper quality.\n"
    "* Assess document quality and text density.\n"
    "* Determine if typewritten.\n"
    "* Assign max 5 descriptive tags as primary_characteristics.\n"
    '* Classify using ONLY one of: "classified_memo", "security_form", "personnel_record", '
    '"operations_roster", "cover_notification", "clearance_request", "administrative_memo", '
    '"historical_record", "field_report"\n'
    '* Rate OCR difficulty as "simple", "average", or "complex".\n'
    "* Return ONLY valid JSON, no markdown, no explanation.\n"
    "</instructions>\n"
    "<output-format>\n"
    '{"includes_handwriting": boolean, "has_shadowy_background": boolean, "document_quality": string, '
    '"text_density": string, "has_stamps": boolean, "has_redactions": boolean, "has_forms": boolean, '
    '"has_tables": boolean, "is_typewritten": boolean, "paper_condition": string, '
    '"primary_characteristics": array, "document_type": string, "ocr_difficulty": string}\n'
    "</output-format>"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_checkpoint() -> set:
    if CHECKPOINT_FILE.exists():
        try:
            data = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
            return {(entry[0], int(entry[1])) for entry in data.get("processed", [])}
        except Exception as exc:
            print(f"[WARNING] Could not load checkpoint: {exc}")
    return set()


def save_checkpoint(processed: set) -> None:
    CHECKPOINT_FILE.write_text(
        json.dumps({"processed": [[fn, pg] for fn, pg in sorted(processed)]}, indent=2),
        encoding="utf-8",
    )


def pil_to_jpeg_bytes(pil_img, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return buf.getvalue()


def extract_json_from_text(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        return json.loads(brace.group(0))
    raise json.JSONDecodeError("No JSON object found in model response", text, 0)


def groq_classify_page(img_b64: str) -> dict:
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    },
                    {"type": "text", "text": GROQ_PROMPT},
                ],
            }
        ],
        temperature=0.1,
        max_tokens=512,
    )
    raw = response.choices[0].message.content.strip()
    return extract_json_from_text(raw)


def google_ocr_page(img_bytes: bytes) -> str:
    image = vision.Image(content=img_bytes)
    resp = vision_client.document_text_detection(image=image)
    if resp.error and resp.error.message:
        raise RuntimeError(f"Vision API error: {resp.error.message}")
    return resp.full_text_annotation.text if resp.full_text_annotation else ""


def build_row(file_id, num_pages, page_num, filename, groq_result, content):
    return {
        "file_id": file_id,
        "number_of_pages": num_pages,
        "page_number": page_num,
        "filename": filename,
        "document_type": groq_result.get("document_type", ""),
        "ocr_difficulty": groq_result.get("ocr_difficulty", ""),
        "includes_handwriting": groq_result.get("includes_handwriting", ""),
        "has_shadowy_background": groq_result.get("has_shadowy_background", ""),
        "document_quality": groq_result.get("document_quality", ""),
        "text_density": groq_result.get("text_density", ""),
        "has_stamps": groq_result.get("has_stamps", ""),
        "has_redactions": groq_result.get("has_redactions", ""),
        "has_forms": groq_result.get("has_forms", ""),
        "has_tables": groq_result.get("has_tables", ""),
        "is_typewritten": groq_result.get("is_typewritten", ""),
        "paper_condition": groq_result.get("paper_condition", ""),
        "primary_characteristics": json.dumps(groq_result.get("primary_characteristics", [])),
        "content": content,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ids = [
        line.strip()
        for line in MISSING_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"[INFO] {len(ids)} file IDs loaded from {MISSING_FILE.name}")

    processed: set = load_checkpoint()
    print(f"[INFO] {len(processed)} pages already in checkpoint — will skip these")

    rows: list[dict] = []
    if OUTPUT_CSV.exists():
        try:
            existing_df = pd.read_csv(OUTPUT_CSV)
            rows = existing_df.to_dict("records")
            print(f"[INFO] {len(rows)} existing rows loaded from {OUTPUT_CSV.name}")
        except Exception as exc:
            print(f"[WARNING] Could not read existing CSV: {exc}")

    pages_since_save = 0

    for doc_idx, file_id in enumerate(ids, start=1):
        filename = f"{file_id}.pdf"
        pdf_path = PDF_DIR / filename

        if not pdf_path.exists():
            print(f"[{doc_idx:2d}/{len(ids)}] MISSING  {filename}")
            continue

        try:
            info = pdfinfo_from_path(str(pdf_path))
            num_pages = int(info["Pages"])
        except Exception as exc:
            print(f"[{doc_idx:2d}/{len(ids)}] ERROR getting page count for {filename}: {exc}")
            continue

        print(f"[{doc_idx:2d}/{len(ids)}] {filename}  ({num_pages} pages)")

        for page_num in range(1, num_pages + 1):
            key = (filename, page_num)

            if key in processed:
                continue

            try:
                images = convert_from_path(
                    str(pdf_path), dpi=100, first_page=page_num, last_page=page_num
                )
                pil_img = images[0]

                img_bytes = pil_to_jpeg_bytes(pil_img, quality=85)
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")

                groq_result = groq_classify_page(img_b64)

                content = google_ocr_page(img_bytes)

                row = build_row(file_id, num_pages, page_num, filename, groq_result, content)
                rows.append(row)
                processed.add(key)
                pages_since_save += 1

                print(
                    f"  [OK]  p{page_num}/{num_pages} "
                    f"→ {groq_result.get('document_type', '?'):25s} "
                    f"| {groq_result.get('ocr_difficulty', '?')}"
                )

                if pages_since_save % AUTOSAVE_EVERY == 0:
                    df_save = pd.DataFrame(rows, columns=CSV_COLUMNS)
                    df_save.to_csv(OUTPUT_CSV, index=False)
                    save_checkpoint(processed)
                    print(f"  [SAVE] autosaved at {len(rows)} total rows")

            except Exception as exc:
                print(f"  [ERROR] {filename} p{page_num}: {exc}")
                traceback.print_exc()
                continue

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df.to_csv(OUTPUT_CSV, index=False)
    save_checkpoint(processed)

    print(f"\n[DONE] CSV saved → {OUTPUT_CSV}")
    print(f"Shape: {df.shape}")
    print("\ndocument_type value_counts:")
    print(df["document_type"].value_counts())


if __name__ == "__main__":
    main()
