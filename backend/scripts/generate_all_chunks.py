#!/usr/bin/env python3
"""
Regenerate chunks for all documents in the DB.
Run:
    python3 backend/scripts/generate_all_chunks.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from db_utils import get_all_documents, update_document_content

fixed = 0
errors = 0

for doc in get_all_documents():
    doc_id = doc['id']
    content = doc['content'] or ''
    try:
        ok = update_document_content(doc_id, content)
        if ok:
            print(f"Regenerated chunks for document id={doc_id} filename={doc['filename']}")
            fixed += 1
        else:
            print(f"Failed to regenerate for id={doc_id}")
            errors += 1
    except Exception as e:
        print(f"Error regenerating for id={doc_id}: {e}")
        errors += 1

print(f"Done. regenerated={fixed}, errors={errors}")
