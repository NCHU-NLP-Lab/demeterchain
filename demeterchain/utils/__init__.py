import importlib
from typing import Any

_module_lookup = {
    "Document": "demeterchain.utils.data",
    "Answer": "demeterchain.utils.data",
    "QAResult": "demeterchain.utils.data",
    "QAModelConfig": "demeterchain.utils.config",
    "QAConfig": "demeterchain.utils.config",
    "PromptTemplate": "demeterchain.utils.template",
    "MessageTemplate": "demeterchain.utils.template",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = list(_module_lookup.keys())