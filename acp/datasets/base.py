from abc import ABC, abstractmethod
from typing import Any, Tuple

from pydantic import BaseModel

from acp.samples.base import BaseSample


class BaseDataset(BaseModel, ABC):
    @property
    @abstractmethod
    def size(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[BaseSample, Any]:
        pass

    def __len__(self) -> int:
        return self.size
