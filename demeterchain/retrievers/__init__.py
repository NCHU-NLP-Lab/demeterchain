import importlib
from typing import Any

_module_lookup = {
    "BaseRetriever": "demeterchain.retrievers.base",
    "RankBM25Retriever": "demeterchain.retrievers.rank_bm25",
    "PyseriniBM25Retriever": "demeterchain.retrievers.pyserini_bm25",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = list(_module_lookup.keys())