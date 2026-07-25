#!/usr/bin/env python3
"""
Script to reprocess uploaded files in uploads/ and update document content in SQLite.
Run from the repo root:

    python3 backend/scripts/reprocess_uploads.py

"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from db_utils import get_all_documents, update_document_content
from knowledge_ingester import _read_local_file

BASE = Path(__file__).resolve().parents[1]
UPLOADS = BASE / 'uploads'

fixed = 0
skipped = 0
errors = 0

for doc in get_all_documents():
    filename = doc['filename']
    content = doc['content'] or ''
    # Heuristic: if content looks like PDF header or contains binary stream markers
    if content.strip().startswith('%PDF') or '\nstream' in content[:200] or 'FlateDecode' in content[:200]:
        file_path = UPLOADS / filename
        if not file_path.exists():
            print(f"Skipping {filename}: file not found in uploads/")
            skipped += 1
            continue
        try:
            new_content = _read_local_file(str(file_path))
            if new_content and new_content.strip():
                ok = update_document_content(doc['id'], new_content)
                if ok:
                    print(f"Reprocessed and updated: {filename}")
                    fixed += 1
                else:
                    print(f"Failed to update DB for: {filename}")
                    errors += 1
            else:
                print(f"No extracted text for: {filename}")
                skipped += 1
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            errors += 1
    else:
        # Not obviously binary-like; skip
        skipped += 1

print(f"Done. fixed={fixed}, skipped={skipped}, errors={errors}")
