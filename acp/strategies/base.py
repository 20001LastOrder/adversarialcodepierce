from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, PrivateAttr

if TYPE_CHECKING:
    from acp.samples.base import BaseSample


class BaseState(ABC, BaseModel):
    score: float = Field(default=0)

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass


class Candidate(BaseModel):
    content: str
    score: float = 0.0


class BaseSearchStrategy(BaseModel, ABC):
    candidates: list[Candidate]
    search_budget: int = 50
    stop_threshold: float = 0
    _current_iteration = 0
    _initial_state: BaseState = None
    _current_state: BaseState = None
    _best_state: BaseState = None
    _visited_states: set[BaseState] = PrivateAttr(default_factory=set)

    @abstractmethod
    def initiate_state(self, sample: BaseSample, original_metric: float) -> BaseState:
        pass

    @abstractmethod
    def next_state(self) -> BaseState:
        pass

    @abstractmethod
    def visit_state(self, state: BaseState, new_metric: float):
        pass

    def should_stop(self):
        """
        Check if the search should stop. The search should stop if the current iteration is greater than the search budget
        or the difference between the initial state and the best state is greater than the stop threshold (e.g. the metric value
        is decreased).
        """
        return (
            self._current_iteration >= self.search_budget
            or self._initial_state.score - self._best_state.score > self.stop_threshold
        )
