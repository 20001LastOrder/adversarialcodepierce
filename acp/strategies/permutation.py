import numpy as np

from acp.samples.base import BaseSample
from acp.strategies.base import BaseSearchStrategy
from acp.strategies.states import VariableState


class PermutationSearch(BaseSearchStrategy):
    """
    Search for an adversarial example based on permuting the variables in
    the code. Note that repeated permutations are allowed.
    """

    random_seed: int = 42

    def __init__(self, **kwargs):
        kwargs["candidates"] = []
        super().__init__(**kwargs)
        np.random.seed(self.random_seed)

    def initiate_state(self, sample: BaseSample, original_metric: float):
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

    def next_state(self):
        self._current_iteration += 1

        state = self._current_state.model_copy(deep=True)

        state.variables = np.random.permutation(state.variables).tolist()

        return state

    def visit_state(self, state: VariableState, new_metric: float):
        state.score = new_metric

        if state.score < self._best_state.score:
            self._best_state = state

        self._current_state = state
