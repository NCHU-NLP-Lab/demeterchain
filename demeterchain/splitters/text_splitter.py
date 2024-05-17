import copy
import logging
import re
from typing import List, Iterator, Optional
from demeterchain.utils import Document

logger = logging.getLogger(__name__)


class TextSplitter(object):
    """
    A class for splitting a list of Document/text into chunks with specified overlap.

    Examples:

        .. code-block:: python

            from demeterchain.loaders import TextLoader
            from demeterchain.splitters import TextSplitter

            # Load text files
            loader = TextLoader('/path/to/directory')
            documents = loader.load()

            # split documents
            text_splitter = TextSplitter(chunk_size=512)
            split_docs = text_splitter.split_documents(documents)

            # split documents with a separator of "\n"
            text_splitter = TextSplitter(separator="\n",chunk_size=512, chunk_overlap=128)
            split_docs = text_splitter.split_documents(documents)
    """
    def __init__(self, separator: Optional[str] = None, chunk_size: int = 512, chunk_overlap: int = 0):
        """
            Args:
            separator (str, optional): The separator string to split the text on. Defaults to None.
            chunk_size (int): The size of each chunk to be created. Defaults to 512.
            chunk_overlap (int): The amount of overlap between consecutive chunks. Defaults to 0.

        """
        self.separator = separator if separator != None else ""
        self.chunk_size = chunk_size if chunk_size >= 0 else 0
        self.chunk_overlap = chunk_overlap if chunk_size > chunk_overlap and chunk_overlap >= 0 else 0
    
    def split_documents(self, documents: Iterator[Document]) -> List[Document]:
        new_documents = []
        for document in documents:
            chunks = self._split_text(document.page_content)
            for chunk in chunks:
                page_content = chunk
                metadata = copy.deepcopy(document.metadata)
                new_doc = Document(page_content=page_content, metadata=metadata)

                new_documents.append(new_doc)
        return new_documents

    def _split_text(self, text: str) -> List[str]:
        splits = re.split(self.separator, text)
        split_lengths = [len(split) for split in splits]
        separator_len = len(self.separator)     

        chunks = []
        start, total = 0, -separator_len
        for i, split_len in enumerate(split_lengths):
            if total + separator_len + split_len > self.chunk_size:
                if total > self.chunk_size:
                    logger.warning(
                        f"Created a chunk of size {total}, "
                        f"which is longer than the specified {self.chunk_size}"
                    )

                if total > 0:
                    chunk = self.separator.join(splits[start:i])
                    chunks.append(chunk)

                    while total > self.chunk_overlap or (
                        total + separator_len + split_len > self.chunk_size
                        and total > 0
                    ):
                        total -= separator_len + split_lengths[start]
                        start += 1


            total += separator_len + split_len  
        chunk = self.separator.join(splits[start:])
        chunks.append(chunk)   
        return chunks
