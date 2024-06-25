import numpy as np

from acp.samples.base import BaseSample
from acp.strategies.base import BaseSearchStrategy
from acp.strategies.states import VariableState


class RandomSearch(BaseSearchStrategy):
    random_seed: int = 42

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        np.random.seed(self.random_seed)

    def initiate_state(
        self, sample: BaseSample, original_metric: float
    ) -> VariableState:
        variables = sample.variables

        variable_names = [variable.name for variable in variables]
        variable_locations = [variable.locations for variable in variables]

        self._initial_state = VariableState(
            variables=variable_names,
            locations=variable_locations,
            score=original_metric,
        )
        self._best_state = self._initial_state
        self._current_state = self._initial_state

        return self._initial_state

    def next_state(self) -> VariableState:
        self._current_iteration += 1
        self._visited_states.add(self._current_state)

        state = self._current_state.model_copy(deep=True)

        if len(state.variables) == 0:
            return state

        next_variable_location = np.random.randint(0, len(state.variables))

        while state in self._visited_states:
            next_candidate_idx = np.random.randint(0, len(self.candidates))
            next_candidate = self.candidates[next_candidate_idx]
            state.variables[next_variable_location] = next_candidate.content

        return state

    def visit_state(self, state: VariableState, new_metric: float):
        state.score = new_metric

        if state.score <= self._best_state.score:
            self._best_state = state

        self._current_state = state
