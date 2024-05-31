from loguru import logger
from pydantic import BaseModel
from tqdm import tqdm

from acp.samples.base import BaseSample
from acp.strategies.base import BaseSearchStrategy


class AdversarialSearch(BaseModel):

    def search(
        self,
        sample: BaseSample,
        predictor: callable,
        search_strategy: BaseSearchStrategy,
        metric: callable,
        verbose: bool = False,
    ):
        original_prediction = predictor(sample)
        original_metric = metric(original_prediction)

        initial_state = search_strategy.initiate_state(
            sample=sample, original_metric=original_metric
        )

        if verbose:
            pbar = tqdm(total=search_strategy.search_budget, leave=False)
        logger.debug(initial_state.variables)

        while not search_strategy.should_stop():
            state = search_strategy.next_state()
            logger.debug(state.variables)
            new_sample = sample.model_copy(deep=True)
            new_sample.update_from_variables(state)

            new_prediction = predictor(new_sample)
            new_metric = metric(new_prediction)

            search_strategy.visit_state(state, new_metric)

            if verbose:
                pbar.update(1)

        return (
            search_strategy._best_state,
            initial_state.score - search_strategy._best_state.score,
        )
