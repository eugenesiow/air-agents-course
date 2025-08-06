import numpy as np
from typing import Any, Dict, List

class InMemoryVectorStore:
    """
    A class to manage an in-memory vector store with a NumPy database.
    """

    def __init__(self, documents: Dict[str, Dict[str, Any]], vectors: Dict[str, np.ndarray]):
        """
        Initializes the in-memory vector store.

        Args:
            documents (Dict[str, Dict[str, Any]]): A dictionary of documents to be stored.
                                                  Example: {'doc1': {'text': '...'}, 'doc2': {'text': '...'}}
            vectors (Dict[str, np.ndarray]): A dictionary mapping document IDs to their corresponding numpy vectors.
        """
        self.documents = documents
        self.vectors = vectors

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs a vector search on the in-memory database.

        Args:
            query_vector (np.ndarray): The vector representation of the query.
            top_k (int): The number of top results to return.

        Returns:
            List[Dict[str, Any]]: A list of the top k most similar documents.
        """
        if not self.vectors:
            return []
            
        # Calculate cosine similarity between the query vector and all document vectors
        similarities = {
            doc_id: np.dot(query_vector, doc_vector.T) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector))
            for doc_id, doc_vector in self.vectors.items()
        }

        # Sort the documents by similarity and get the top k results
        sorted_docs = sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:top_k]

        # Format the results
        results = []
        for doc_id, score in sorted_docs:
            # The 'score' is a numpy array with a single value, e.g., array([0.87]), so we access it with [0]
            results.append(
                {
                    "id": doc_id,
                    "score": score[0],
                    "content": self.documents[doc_id],
                }
            )
        return results
