from __future__ import annotations

import json
import logging
import subprocess
import importlib.util
from pathlib import Path
from typing import Any, List, Iterable, Optional, Dict, Callable
from demeterchain.utils import Document
from demeterchain.retrievers import BaseRetriever

logger = logging.getLogger(__name__)


class PyseriniBM25Retriever(BaseRetriever):
    """`
    BM25` retriever created through Pyserini.
    
    Examples:

        .. code-block:: python

            from demeterchain.loaders import TextLoader
            from demeterchain.retrievers import PyseriniBM25Retriever

            # Load text files
            loader = TextLoader('/path/to/directory')
            documents = loader.load()

            # build and load retriever
            retriever = PyseriniBM25Retriever.from_documents(documents, language = "zh", savepath = "pyserini_bm25_index")

            # direct load retriever
            retriever = PyseriniBM25Retriever.load("pyserini_bm25_index")

            # retrieve top-k documents
            query = "我是誰"
            print(retriever.invoke(query, k=5))
    """

    retriever: Any
    """ BM25 retriever."""
    index_filepath: str
    """ Pyserini index"""
    language: Optional[str]
    """ Pyserini index language"""

    DOCS_SAVE_NAME = "docs/docs.jsonl"
    INDEX_SAVE_NAME = "index"
    CONFIG_SAVE_NAME = "config.json"
    JAVA_ALLOWED_VERSIONS = [11]
    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        metadatas: Optional[Iterable[dict]] = None,
        language: Optional[str] = None,
        savepath: str = "pyserini_bm25_index",
    ) -> PyseriniBM25Retriever:
        """
        Creates a retriever object from given texts and metadata.

        Args:
            texts (Iterable[str]): A iterable of texts to retrieve data from.
            metadatas (Iterable[dict]): A iterable of metadata corresponding to the texts.
            language (str): The language to be used for processing the texts.
            savepath (str): The path where the retriever will be automatically saved after creation. Defaults to "pyserini_bm25_index".

        Returns:
            Retriever: A retriever object created from the provided texts and metadata.
        """
        if not cls.check_jdk_version(cls.JAVA_ALLOWED_VERSIONS):
            raise RuntimeError(
                "Java is not installed or configured correctly.\n"
                "Please install Java using the following commands:\n"
                "`sudo apt-get update && sudo apt-get install openjdk-11-jdk`"
            ) 

        metadatas = metadatas or ({} for _ in texts)
        directory = Path(savepath)

        # build jsonl for pyserini
        docs_filename = directory.joinpath(cls.DOCS_SAVE_NAME)
        docs_filename.parent.mkdir(parents=True, exist_ok=True)
        cls.build_jsonl_data(texts, metadatas, docs_filename)

        # build pyserini index
        index_dir = directory.joinpath(cls.INDEX_SAVE_NAME)
        cls.build_bm25_index(docs_filename.parent, index_dir, language)

        # build config
        config = {
            "class_name" : cls.__name__,
            "docs_size" : len(texts),
            "language" : language,
        }
        config_filename = directory.joinpath(cls.CONFIG_SAVE_NAME)
        with open(config_filename, 'w') as f:
            json.dump(config, f, indent=4)

        return cls.load(savepath)

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        language: Optional[str] = "zh",
        savepath: str = "pyserini_bm25_index",
    ) -> PyseriniBM25Retriever:
        """
        Creates a retriever object from given documents.

        Args:
            documents (Iterable[str]): A iterable of documents to retrieve data from.
            language (str): The language to be used for processing the texts.
            savepath (str): The path where the retriever will be automatically saved after creation. Defaults to "pyserini_bm25_index".

        Returns:
            Retriever: A retriever object created from the provided texts and metadata.
        """
        texts, metadatas = zip(*((doc.page_content, doc.metadata) for doc in documents))
        return cls.from_texts(
            texts=texts,
            metadatas=metadatas,
            language=language,
            savepath=savepath,
        )
    
    @classmethod
    def load(cls, filepath: str) -> PyseriniBM25Retriever:
        """
        load retriever from filepath.
        
        Args:
            filepath: The filepath from which the retriever will be loaded.

        Returns:
            PyseriniBM25Retriever
        """
        directory = Path(filepath)
        if not directory.exists():
            raise FileNotFoundError(f"Directory {filepath} not found")

        # load config
        config_filename = directory.joinpath(cls.CONFIG_SAVE_NAME)
        if config_filename.exists():
            with open(config_filename, "r") as file:
                data = json.load(file)
                language = data["language"]
        else:
            raise FileNotFoundError(f"Config file {config_filename} not found")

        # load Pyserini retriever
        try:
            from pyserini.search.lucene import LuceneSearcher
        except ImportError:
            raise ImportError(
                "Could not import pyserini, please install with `pip install "
                "pyserini==0.22.1 faiss-cpu==1.7.4`."
            )
        index_dir = directory.joinpath(cls.INDEX_SAVE_NAME)
        searcher = LuceneSearcher(str(index_dir))
        if language != None:
            searcher.set_language(language)
        return cls(retriever=searcher, index_filepath=filepath)

    def save(self, *args, **kwargs):
        """
        PyseriniBM25Retriever  does not support save method.
        """
        logger.warning(
            "PyseriniBM25Retriever does not support save method, "
            f"the index has been automatically saved in `{self.index_filepath}` when built")
    
    def get_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
        hits = self.retriever.search(query, k)
        candicate_context = []
        return_docs = []
        for i in hits:
            raw_data = json.loads(i.raw)
            page_content = raw_data["contents"]
            metadata = raw_data["metadata"] if "metadata" in raw_data else {}
            return_docs.append(Document(page_content=page_content, metadata=metadata))
        return return_docs


    @staticmethod
    def build_jsonl_data(texts, metadatas, docs_filename):
        try:
            with open(docs_filename, 'w', encoding='utf-8') as f:
                for i, (text, metadata) in enumerate(zip(texts, metadatas)):
                    index_data = {
                        "id": i,
                        "metadata": metadata,
                        "contents": text
                    }
                    json_data = json.dumps(index_data, ensure_ascii=False)
                    f.write(json_data + '\n')
        except IOError as e:
            raise IOError(f"An error occurred while writing to the file: {e}")

    @staticmethod
    def build_bm25_index(docs_dir: str, index_dir: str, language: Optional[str] = "zh"):
        if not importlib.util.find_spec("pyserini"):
            raise ModuleNotFoundError(
                "Could not found pyserini, please install with `pip install "
                "pyserini==0.22.1 faiss-cpu==1.7.4`."
            )
        try:
            command = [
                'python', '-m', 'pyserini.index.lucene',
                '--collection', 'JsonCollection',
                '--input', docs_dir,
                '--index', index_dir,
                '--generator', 'DefaultLuceneDocumentGenerator',
                '--threads', '1',
                '--storePositions', '--storeDocvectors', '--storeRaw'
            ]
            if language != None:
                command.extend(['--language', language])
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error: Unable to create index. Error: \n{e}")
    
    @staticmethod
    def check_jdk_version(allowed_versions):
        try:
            output = subprocess.check_output(['java', '-version'], stderr=subprocess.STDOUT)
            output_str = output.decode('utf-8')
        except subprocess.CalledProcessError:
            return False
        
        return any(f"version \"{version}" in output_str for version in allowed_versions)

