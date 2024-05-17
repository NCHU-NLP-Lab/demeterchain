from abc import ABC, abstractmethod
from typing import Any, List
from demeterchain.utils import Document

class BaseRetriever(ABC):
    """Abstract base class for a Document retrieval system.

    A retrieval system is defined as something that can take string queries and return
    the most 'relevant' Documents from some source.

    Usage:

    Use `invoke` method to retrieve documents.
    """
    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    def invoke(self, query: str, **kwargs: Any) -> List[Document]:
        """
        Get documents relevant to a query.
        
        Args:
            query: String to find relevant documents
        Returns:
            List of relevant documents
        """
        return self.get_relevant_documents(
            query,
            **kwargs,
        )
    
    @abstractmethod
    def get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        """
        Get documents relevant to a query.
        
        Args:
            query: String to find relevant documents
        Returns:
            List of relevant documents
        """

    @abstractmethod
    def save(self, filepath, **kwargs: Any):
        """
        save retriever to filepath.
        
        Args:
            filepath: The filepath where the retriever will be saved.
        """

    @classmethod
    @abstractmethod
    def load(cls, filepath, **kwargs: Any):
        """
        load retriever from filepath.
        
        Args:
            filepath: The filepath from which the retriever will be loaded.

        Returns:
            Retriever: The loaded retriever object.
        """