import os, time, torch, gradio as gr
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from datetime import datetime
import random
import string

from transformers import AutoModel, AutoTokenizer
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_name = 'deepseek-ai/DeepSeek-OCR'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
model = model.eval().cuda().to(torch.bfloat16)

# prompt = "<image>\nFree OCR. "
prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = 'your_image.jpg'
output_path = 'your/output/dir'

# infer(self, tokenizer, prompt='', image_file='', output_path = ' ', base_size = 1024, image_size = 640, crop_mode = True, test_compress = False, save_results = False):

# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False

# Gundam: base_size = 1024, image_size = 640, crop_mode = True

res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 1024, image_size = 640, crop_mode=True, save_results = True, test_compress = True)

import os
import pytesseract
import requests
from pdf2image import convert_from_path
from rapidfuzz import fuzz
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
PDF_DIR = "pdfs"
GT_DIR = "ground_truth"
OUT_TESS = "outputs/tesseract"
OUT_DEEPSEEK = "outputs/deepseek"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_URL = "https://api.deepseek.com/ocr"   # Example placeholder

os.makedirs(OUT_TESS, exist_ok=True)
os.makedirs(OUT_DEEPSEEK, exist_ok=True)


# -----------------------------
# 1. RUN TESSERACT
# -----------------------------
def ocr_tesseract_page(image):
    return pytesseract.image_to_string(image)


def run_tesseract(pdf_path, save_path):
    pages = convert_from_path(pdf_path)
    full_text = ""

    for i, page in enumerate(pages):
        text = ocr_tesseract_page(page)
        full_text += f"\n\n--- PAGE {i+1} ---\n{text}"

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(full_text)


# -----------------------------
# 2. RUN DEEPSEEK OCR (API CALL)
# -----------------------------
def deepseek_ocr(image):
    """
    Convert PIL Image → bytes → API request.
    Replace endpoint below with correct DeepSeek OCR endpoint.
    """
    import io
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/octet-stream"
    }

    response = requests.post(DEEPSEEK_URL, headers=headers, data=img_bytes)
    response.raise_for_status()
    return response.json().get("text", "")


def run_deepseek(pdf_path, save_path):
    pages = convert_from_path(pdf_path)
    full_text = ""

    for i, page in enumerate(pages):
        txt = deepseek_ocr(page)
        full_text += f"\n\n--- PAGE {i+1} ---\n{txt}"

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(full_text)


# -----------------------------
# 3. ACCURACY COMPARISON
# -----------------------------
def calculate_accuracy(gt_text, test_text):
    """
    fuzzy character + fuzzy word similarity
    """
    char_score = fuzz.ratio(gt_text, test_text)
    word_score = fuzz.token_sort_ratio(gt_text, test_text)
    return char_score, word_score


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def run_pipeline():
    pdf_files = list(Path(PDF_DIR).glob("*.pdf"))

    results = []

    for pdf in pdf_files:
        name = pdf.stem
        print(f"Processing {name} ...")

        tess_out = f"{OUT_TESS}/{name}.txt"
        deep_out = f"{OUT_DEEPSEEK}/{name}.txt"
        gt_file = Path(GT_DIR) / f"{name}.txt"

        # --- run OCRs
        run_tesseract(pdf, tess_out)
        run_deepseek(pdf, deep_out)

        # --- compare
        if gt_file.exists():
            gt = gt_file.read_text(encoding="utf-8")
            tess_txt = Path(tess_out).read_text(encoding="utf-8")
            deep_txt = Path(deep_out).read_text(encoding="utf-8")

            tess_scores = calculate_accuracy(gt, tess_txt)
            deep_scores = calculate_accuracy(gt, deep_txt)

            results.append({
                "file": name,
                "tesseract_char_acc": tess_scores[0],
                "tesseract_word_acc": tess_scores[1],
                "deepseek_char_acc": deep_scores[0],
                "deepseek_word_acc": deep_scores[1],
            })

    # Print table-like output
    print("\n\n=== OCR ACCURACY RESULTS ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    run_pipeline()
