import os
import pickle
import pdfplumber
import numpy as np
import pandas as pd
from typing import Dict, List

def extract_text_with_pdfplumber(pdf_path: str) -> str:
    """
    Extracts plain text from all pages of a PDF using pdfplumber.
    This provides better layout preservation than PyPDF2.
    """
    print(f"Extracting text from {pdf_path} using pdfplumber...")
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
    print("Text extraction successful.")
    return full_text

def extract_tables_with_pdfplumber(pdf_path: str) -> str:
    """
    Extracts tables from a PDF using pdfplumber and returns them as an HTML string.
    This method does not have external dependencies like Ghostscript.
    """
    print(f"Extracting tables from {pdf_path} using pdfplumber...")
    html_tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                if not tables:
                    continue
                
                print(f"Found {len(tables)} tables on page {i + 1}.")
                for table in tables:
                    # Convert list of lists to a pandas DataFrame
                    # A basic assumption is that the first row is the header
                    if table:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        html_tables.append(df.to_html(index=False, classes='table table-striped', justify='left'))

        if not html_tables:
            print("No tables were extracted from the document.")
            return ""

        print(f"Successfully extracted {len(html_tables)} tables in total.")
        return "\n<br/>\n".join(html_tables)
    except Exception as e:
        print(f"Could not extract tables with pdfplumber. Error: {e}")
        return ""

def chunk_text(full_text: str, chunk_size: int = 512, overlap: int = 100) -> Dict[str, Dict[str, str]]:
    """
    Splits text into chunks and returns a document dictionary.
    """
    print("Chunking extracted text...")
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunks.append(full_text[start:end])
        start += chunk_size - overlap

    documents = {f"chunk_{i}": {"text": chunk} for i, chunk in enumerate(chunks)}
    print(f"Created {len(documents)} document chunks.")
    return documents


def generate_document_embeddings(documents: Dict[str, Dict[str, str]], embedding_client) -> Dict[str, np.ndarray]:
    """
    Generates embeddings for a dictionary of document chunks.
    """
    print("Generating embeddings for document chunks...")
    texts_to_embed = [doc["text"] for doc in documents.values()]
    if not texts_to_embed:
        return {}
    
    response = embedding_client.create(
        input=texts_to_embed,
        encoding_format="float",
        model="nvidia/nv-embedqa-mistral-7b-v2",
        extra_body={
            "input_type": "query",
            "truncate": "NONE",
        },
    )
    
    vectors = {doc_id: np.array(embedding.embedding, dtype=np.float32) for doc_id, embedding in zip(documents.keys(), response.data)}
    print(f"Successfully generated {len(vectors)} embeddings.")
    return vectors

def cache_embeddings(documents: Dict, vectors: Dict, cache_path: str):
    """
    Saves documents and their embeddings to a pickle file.
    """
    print(f"Saving documents and embeddings to cache: {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump({"documents": documents, "vectors": vectors}, f)

def load_cached_embeddings(cache_path: str) -> Dict:
    """
    Loads documents and embeddings from a pickle file.
    """
    print(f"Loading documents and embeddings from cache: {cache_path}")
    with open(cache_path, 'rb') as f:
        return pickle.load(f)
