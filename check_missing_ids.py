from pathlib import Path

pdf_dir = Path.home() / "Desktop" / "Thesis" / "jfk_pdfs"
missing_file = Path.home() / "Desktop" / "Thesis" / "missing_file_ids.txt"

ids = [line.strip() for line in missing_file.read_text(encoding="utf-8").splitlines() if line.strip()]

# Build a case-insensitive set of available filenames
available = {p.name.lower(): p for p in pdf_dir.glob("*") if p.is_file()}

found = []
not_found = []

for doc_id in ids:
    target = f"{doc_id}.pdf".lower()
    if target in available:
        found.append(doc_id)
    else:
        # also try searching recursively just in case
        matches = list(pdf_dir.rglob(f"{doc_id}.pdf"))
        matches += list(pdf_dir.rglob(f"{doc_id}.PDF"))
        if matches:
            found.append(doc_id)
        else:
            not_found.append(doc_id)

print("PDF DIR:", pdf_dir)
print("Missing IDs in file:", len(ids))
print("FOUND PDFs:", len(found))
print("NOT FOUND:", len(not_found))

if not_found:
    print("\n--- NOT FOUND IDs ---")
    for x in not_found:
        print(x)
