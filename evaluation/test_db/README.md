# Test Database Directory

This directory contains the test database and FAISS index files used for SPICED evaluation.

## Files

- `test_spiced.db` - SQLite database containing 1954 SPICED articles
- `test_faiss.index` - FAISS index for semantic search
- `test_faiss_metadata.pkl` - Metadata mapping for FAISS index

## Usage

The test database is automatically created and used by the evaluation pipeline when running:

```bash
python evaluation/evaluation_pipeline.py
```

## Regeneration

To regenerate the test database:

```bash
python evaluation/test_database.py
```

## Notes

- The test database contains SPICED articles from `spiced.csv`
- Each SPICED pair becomes two articles (text_1 and text_2)
- The FAISS index is rebuilt automatically when needed
- These files are separate from the main system database to avoid conflicts
