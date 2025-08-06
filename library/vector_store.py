import numpy as np
from typing import Any, Dict, List

class InMemoryVectorStore:
    """
    A class to manage an in-memory vector store with a NumPy database.
    """

    def __init__(self, documents: Dict[str, Dict[str, Any]], vectors: Dict[str, np.ndarray]):
        """
        Initializes the in-memory vector store.
        """
        self.documents = documents
        self.vectors = vectors

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs a vector search on the in-memory database.
        """
        if not self.vectors:
            return []
            
        similarities = {
            doc_id: np.dot(query_vector, doc_vector.T) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector))
            for doc_id, doc_vector in self.vectors.items()
        }

        sorted_docs = sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:top_k]

        results = []
        for doc_id, score in sorted_docs:
            results.append(
                {
                    "id": doc_id,
                    "score": score[0],
                    "content": self.documents[doc_id],
                }
            )
        return results
