from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def get_answer(self):
        pass

