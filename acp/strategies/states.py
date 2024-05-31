from pydantic import model_validator

from acp.strategies.base import BaseState


class VariableState(BaseState):
    variables: list[str]
    locations: list[list[int]]

    def __hash__(self):
        return hash(tuple(self.variables))

    def __eq__(self, other):
        return self.variables == other.variables


class RandomVectorState(BaseState):
    variables: list[str]
    locations: list[list[int]]
    vectors: list[list[float]] = None

    @model_validator(mode="after")
    def set_vectors(self):
        if self.vectors is None:
            self.vectors = [None for _ in self.variables]

    def __hash__(self):
        return hash(tuple(self.variables))

    def __eq__(self, other):
        # Since the vectors are VERY unlikely to be the same, we can
        # always return False here.
        return False
