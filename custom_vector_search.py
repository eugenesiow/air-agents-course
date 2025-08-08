import asyncio
import os
import pickle
from typing import Any, Dict, List
import PyPDF2
import numpy as np
from air import AIRefinery, DistillerClient, login
from air.utils import async_print
from dotenv import load_dotenv

from numpy_vector_store import InMemoryVectorStore

# Load environment variables from .env file, overriding existing ones
load_dotenv(override=True)

def load_and_chunk_pdf(pdf_path: str, chunk_size: int = 512, overlap: int = 100) -> Dict[str, Dict[str, str]]:
    """
    Loads text from a PDF, splits it into chunks, and returns a document dictionary.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return {}

    print(f"Loading and chunking PDF: {pdf_path}")
    full_text = ""
    with open(pdf_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            full_text += page.extract_text() + "\n"

    # Simple chunking logic
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunks.append(full_text[start:end])
        start += chunk_size - overlap

    documents = {f"chunk_{i}": {"text": chunk} for i, chunk in enumerate(chunks)}
    print(f"Created {len(documents)} document chunks from the PDF.")
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

# --- Main script setup ---

EMBEDDINGS_CACHE_PATH = "embeddings.pickle"

# AI Refinery authentication and client setup
auth = login(
    account=str(os.getenv("ACCOUNT")),
    api_key=str(os.getenv("API_KEY")),
)
base_url = os.getenv("AIREFINERY_ADDRESS", "")
air_client = AIRefinery(**auth.openai(base_url=base_url))
embedding_client = air_client.embeddings
distiller_client = DistillerClient(base_url=base_url)

# Check for cached embeddings and documents
if os.path.exists(EMBEDDINGS_CACHE_PATH):
    print(f"Loading documents and embeddings from cache: {EMBEDDINGS_CACHE_PATH}")
    with open(EMBEDDINGS_CACHE_PATH, 'rb') as f:
        cached_data = pickle.load(f)
        documents_from_pdf = cached_data["documents"]
        document_vectors = cached_data["vectors"]
else:
    print("Cache not found. Processing PDF and generating new embeddings.")
    # Load documents from the PDF
    pdf_path = "data/acn-third-quarter-fiscal-2025-earnings-release.pdf"
    documents_from_pdf = load_and_chunk_pdf(pdf_path)

    # Generate embeddings for the documents
    document_vectors = generate_document_embeddings(documents_from_pdf, embedding_client)
    
    # Cache the documents and their embeddings
    if documents_from_pdf and document_vectors:
        print(f"Saving documents and embeddings to cache: {EMBEDDINGS_CACHE_PATH}")
        with open(EMBEDDINGS_CACHE_PATH, 'wb') as f:
            pickle.dump({"documents": documents_from_pdf, "vectors": document_vectors}, f)

# Initialize the in-memory vector store
vector_store = InMemoryVectorStore(documents_from_pdf, document_vectors)

uuid = os.getenv("UUID", "test_user")
project = "DocumentSearch"


async def custom_in_memory_vector_search(query: str):
    """
    Executor for custom in-memory vector search over the PDF content.
    """
    print(f"Received query for vector search: '{query}'")
    # Generate an embedding for the query
    response = embedding_client.create(
        input=[query],
        model="nvidia/nv-embedqa-mistral-7b-v2",
        encoding_format="float",
        extra_body={
            "input_type": "query",
            "truncate": "NONE",
        },
    )
    query_vector = np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)
    
    print("Searching for relevant documents in the vector store...")
    documents = vector_store.search(query_vector)

    if not documents:
        return [{"result": "There is no relevant document from the PDF.", "score": 0}]

    results = [
        _format_document_result(
            doc["id"], doc, source_weight=1, retriever_name="PDF-NumPy-Retriever"
        )
        for doc in documents
    ]
    return results


def _format_document_result(
    doc_id: Any, doc: Dict[str, Any], source_weight=1, retriever_name=""
) -> Dict[str, Any]:
    """
    Format a single document into a structured result.
    """
    base_score = doc.get("score", 0.0)
    final_score = float(base_score * source_weight)
    content_text = doc.get("content", {}).get("text", "")
    
    # Format the result string to be clean and readable
    formatted_result = f"Source: {retriever_name}\nID: {doc_id}\nContent: {content_text[:300]}..."

    return {"result": formatted_result, "score": final_score}


async def custom_vector_search_test():
    """
    Function for testing the in-memory vector search.
    """
    distiller_client.create_project(
        config_path="custom_vector_search.yaml", project=project
    )

    executor_dict = {
        "Research Agent": {
            "Fiscal Reports Database": custom_in_memory_vector_search,
        }
    }

    async with distiller_client(
        project=project,
        uuid=uuid,
        executor_dict=executor_dict,
    ) as dc:
        queries = [
            "what can you tell me about the financial results?",
            "what were the days services outstanding and how did they compare with previous years?",
        ]
        for query in queries:
            responses = await dc.query(query=query)
            print(f"----\nQuery: {query}")
            async for response in responses:
                await async_print(f"Response: {response['content']}")


if __name__ == "__main__":
    if 'documents_from_pdf' not in locals() or not documents_from_pdf:
        print("Could not load documents from PDF or cache. Aborting test.")
    else:
        asyncio.run(custom_vector_search_test())
