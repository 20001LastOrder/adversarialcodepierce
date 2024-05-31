from abc import ABC, abstractmethod

from pydantic import BaseModel


class BaseMetric(BaseModel, ABC):
    @abstractmethod
    def evaluate(self, *args, **kwargs) -> float:
        pass
