from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING

from pydantic import BaseModel, computed_field, model_validator

if TYPE_CHECKING:
    from acp.strategies.states import VariableState

__all__ = ["BaseSample"]


class Variable(BaseModel):
    name: str
    locations: list[int]


class BaseSample(BaseModel, ABC):
    code: str
    tokenized_code: list[str] = None
    variables: list[Variable] = None

    @model_validator(mode="after")
    def infer_properties(self):
        if self.tokenized_code is None:
            self.tokenized_code = self.tokenize(self.code)
        self.tokenized_code[1] = "<mask>"
        if self.variables is None:
            self.variables = self.create_variables()

    def update_from_variables(self, state: VariableState):
        for variable, new_name in zip(self.variables, state.variables):
            variable.name = new_name

        for variable in self.variables:
            for location in variable.locations:
                self.tokenized_code[location] = variable.name

        self.code = " ".join(self.tokenized_code)

    @computed_field
    @cached_property
    @abstractmethod
    def ast(self) -> object:
        pass

    @abstractmethod
    def create_variables(self) -> list[Variable]:
        pass

    @abstractmethod
    def tokenize(self, code: str) -> list[str]:
        pass
