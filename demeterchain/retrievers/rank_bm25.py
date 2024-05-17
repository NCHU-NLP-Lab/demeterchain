from __future__ import annotations

import pickle
import json
import importlib.util
from pathlib import Path
from typing import Any, List, Iterable, Optional, Dict, Callable
from demeterchain.utils import Document
from demeterchain.retrievers import BaseRetriever

def default_chinese_tokenize(text: str) -> List[str]:
    try:
        import jieba
    except ImportError:
        raise ImportError(
            "Could not import jieba, please install with `pip install "
            "jieba`."
        )
    return list(jieba.cut(text))

class RankBM25Retriever(BaseRetriever):
    """`
    BM25` retriever created through rank_bm25.
    
    Examples:

        .. code-block:: python

            from demeterchain.loaders import TextLoader
            from demeterchain.retrievers import RankBM25Retriever

            # Load text files
            loader = TextLoader('/path/to/directory')
            documents = loader.load()

            # build and load retriever
            retriever = RankBM25Retriever.from_documents(documents)

            # save retriever
            retriever.save("bm25/test")

            # direct load retriever
            retriever = RankBM25Retriever.load("bm25/test")

            # retrieve top-k documents
            query = "我是誰"
            print(retriever.invoke(query, k=5))
    """
    retriever: Any
    """ BM25 retriever."""
    docs: List[Document]
    """ List of documents."""
    preprocess_func: Callable[[str], List[str]]
    """ Preprocessing function to use on the text before BM25 vectorization."""

    RETRIEVER_SAVE_NAME = "retriever.obj"
    DOCS_SAVE_NAME = "docs.obj"
    PREPROCESS_FUNC_SAVE_NAME = "preprocess_func.obj"
    CONFIG_SAVE_NAME = "config.json"

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        metadatas: Optional[Iterable[dict]] = None,
        bm25_params: Optional[Dict[str, Any]] = None,
        preprocess_func: Callable[[str], List[str]] = default_chinese_tokenize,
    ) -> RankBM25Retriever:
        """
        Creates a retriever object from given texts and metadata.

        Args:
            texts (Iterable[str]): A iterable of texts to retrieve data from.
            metadatas (Iterable[dict]): A iterable of metadata corresponding to the texts.
            bm25_params (Optional[Dict[str, Any]]): Parameters used for BM25Okapi.
            preprocess_func (Callable[[str], List[str]]): Pre-processing of data, such as word segmentation, Stemming and Lemmatization. Defaults to default_chinese_tokenize(use jieba for word segmentation).

        Returns:
            Retriever: A retriever object created from the provided texts and metadata.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "Could not import rank_bm25, please install with `pip install "
                "rank_bm25`."
            )

        texts_processed = [preprocess_func(t) for t in texts]
        bm25_params = bm25_params or {}
        retriever = BM25Okapi(texts_processed, **bm25_params)
        metadatas = metadatas or ({} for _ in texts)
        docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
        return cls(
            retriever=retriever, docs=docs, preprocess_func=preprocess_func
        )

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        bm25_params: Optional[Dict[str, Any]] = None,
        preprocess_func: Callable[[str], List[str]] = default_chinese_tokenize,
    ) -> RankBM25Retriever:
        """
        Creates a retriever object from given documents.

        Args:
            documents (Iterable[str]): A iterable of documents to retrieve data from.
            bm25_params (Optional[Dict[str, Any]]): Parameters used for BM25Okapi.
            preprocess_func (Callable[[str], List[str]]): Pre-processing of data, such as word segmentation, Stemming and Lemmatization. Defaults to default_chinese_tokenize(use jieba for word segmentation).

        Returns:
            Retriever: A retriever object created from the provided texts and metadata.
        """
        texts, metadatas = zip(*((doc.page_content, doc.metadata) for doc in documents))
        return cls.from_texts(
            texts=texts,
            bm25_params=bm25_params,
            metadatas=metadatas,
            preprocess_func=preprocess_func,
        )
    @classmethod
    def load(cls, filepath):
        """
        load retriever from filepath.
        
        Args:
            filepath: The filepath from which the retriever will be loaded.

        Returns:
            RankBM25Retriever
        """
        if not importlib.util.find_spec("rank_bm25"):
            raise ModuleNotFoundError(
                "Could not found rank_bm25, please install with `pip install "
                "rank_bm25`."
            )
        
        directory = Path(filepath)
        if not directory.exists():
            raise FileNotFoundError(f"Directory {filepath} not found")
        
        # load datas
        filenames = {
            "retriever" : cls.RETRIEVER_SAVE_NAME,
            "docs" : cls.DOCS_SAVE_NAME,
            "preprocess_func" : cls.PREPROCESS_FUNC_SAVE_NAME
        }
        load_datas ={}
        for data, name in filenames.items():
            filename = directory.joinpath(name)
            if not filename.exists():
                raise FileNotFoundError(f"File {filename} not found")
            with open(filename, 'rb') as f:
                load_datas[data] = pickle.load(f)

        return cls(**load_datas)

    def save(self, filepath):
        """
        save retriever to filepath.
        
        Args:
            filepath: The filepath which the retriever will be saved.
        """
        directory = Path(filepath)
        directory.mkdir(parents=True, exist_ok=True)
 
        # save config
        config = {
            "class_name" : self.__class__.__name__,
            "docs_size" : len(self.docs),
        }
        config_filename = directory.joinpath(self.CONFIG_SAVE_NAME)
        with open(config_filename, 'w') as f:
            json.dump(config, f, indent=4)

        # save datas
        save_files = {
            self.RETRIEVER_SAVE_NAME : self.retriever,
            self.DOCS_SAVE_NAME : self.docs,
            self.PREPROCESS_FUNC_SAVE_NAME : self.preprocess_func
        }
        for save_name, data in save_files.items():
            filename = directory.joinpath(save_name)
            with open(filename, 'wb') as f:
                pickle.dump(data, f)  
    
    def get_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
        processed_query = self.preprocess_func(query)
        return_docs = self.retriever.get_top_n(processed_query, self.docs, n=k)
        return return_docs

