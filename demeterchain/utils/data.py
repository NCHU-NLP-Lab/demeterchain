from typing import Any, Dict, Optional


class Document(object):
    DISPLAY_LEN = 50
    def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
        self.page_content = page_content
        self.metadata = metadata if metadata != None else {}
    
    def __repr__(self):
        content_str = self.page_content if len(self.page_content) < self.DISPLAY_LEN else f"{self.page_content[:self.DISPLAY_LEN]}..."
        return f"Document(page_content={repr(content_str)}, metadata={repr(self.metadata)})"


class Answer(object):
    def __init__(self, answer: str, doc: Optional[Document] = None):
        self.answer = answer
        self.doc = doc
    
    def __repr__(self):
        return f"Answer(answer={repr(self.answer)}, doc={repr(self.doc)})"


class QAResult(object):
    def __init__(self, answers: Dict[str, Document], summary: Optional[str] = None):
        self.summary = summary
        _answers = []
        for ans, doc in answers.items():
            _answers.append(Answer(answer=ans, doc=doc))
        self.answers = _answers
    
    def __repr__(self):
        info = ["QAResult(", f"    summary = {repr(self.summary)},", "    answers = ["]
        for ans in self.answers:
            info.append(f"        {repr(ans)}")
        info.append("    ]")
        info.append(")")
        return "\n".join(info)