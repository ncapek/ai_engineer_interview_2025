#!/usr/bin/env python3
"""Upload documents to the vector store.

This script uploads documents (PDF, DOCX, TXT, MD) to the MongoDB vector store
for later retrieval and question answering.

Usage Examples:
    # Upload a single file
    uv run python scripts/upload_documents.py document.pdf

    # Upload multiple files
    uv run python scripts/upload_documents.py doc1.pdf doc2.docx doc3.txt

    # Upload all supported files from a directory
    uv run python scripts/upload_documents.py sample_documents/

    # Clear existing chunks and upload new documents
    uv run python scripts/upload_documents.py --clear sample_documents/

    # Mix files and directories
    uv run python scripts/upload_documents.py file1.pdf sample_documents/ file2.docx
"""
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from doctalk.embedding import EmbeddingManager
from doctalk.pipeline import IngestionPipeline
from doctalk.storage import VectorStoreManager

load_dotenv()

SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.md', '.docx'}


def get_files_from_path(path: str) -> list[str]:
    """Get list of supported files from a path (file or directory).

    :param path: File path or directory path
    :return: List of file paths
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f'Path not found: {path}')

    if path_obj.is_file():
        if path_obj.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f'Unsupported file type: {path_obj.suffix}')
        return [str(path_obj)]

    if path_obj.is_dir():
        files = set()
        for ext in SUPPORTED_EXTENSIONS:
            files.update(path_obj.glob(f'*{ext}'))
            files.update(path_obj.glob(f'**/*{ext}'))
        return [str(f) for f in files if f.is_file()]

    return []


def main():
    parser = argparse.ArgumentParser(description='Upload documents to the vector store')
    parser.add_argument(
        'paths',
        nargs='+',
        help='File paths or directory paths to upload',
    )
    parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear existing chunks before uploading',
    )

    args = parser.parse_args()

    embedder = EmbeddingManager()
    vector_store = VectorStoreManager(
        mongodb_uri=os.environ['MONGODB_URI'],
        db_name='doctalk',
        collection_name='chunks',
    )
    pipeline = IngestionPipeline(embedder, vector_store)

    if args.clear:
        vector_store.collection.delete_many({})
        print('Cleared existing chunks.')

    all_files = []
    for path in args.paths:
        try:
            files = get_files_from_path(path)
            all_files.extend(files)
        except (FileNotFoundError, ValueError) as e:
            print(f'Error with {path}: {e}')
            continue

    if not all_files:
        print('No supported files found.')
        return

    print(f'Found {len(all_files)} file(s) to upload:')
    for f in all_files:
        print(f'  - {f}')

    print('\nUploading...')
    for file_path in all_files:
        try:
            pipeline.ingest(file_path)
            print(f'✓ {Path(file_path).name}')
        except Exception as e:
            print(f'✗ {Path(file_path).name}: {e}')

    print('\nUpload complete!')


if __name__ == '__main__':
    main()
