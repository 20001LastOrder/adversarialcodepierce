import random

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


class AdversarialSearchPairs(BaseModel):

    def search(
        self,
        sample1: BaseSample,
        sample2: BaseSample,
        predictor: callable,
        search_strategy1: BaseSearchStrategy,
        search_strategy2: BaseSearchStrategy,
        metric: callable,
        verbose: bool = False,
    ):
        original_prediction = predictor(sample1, sample2)
        original_metric = metric(original_prediction)
        # print(f"initial metric{original_metric}")

        initial_state1 = search_strategy1.initiate_state(
            sample=sample1, original_metric=original_metric
        )
        initial_state2 = search_strategy2.initiate_state(
            sample=sample2, original_metric=original_metric
        )

        if verbose:
            pbar = tqdm(
                total=search_strategy1.search_budget + search_strategy2.search_budget,
                leave=False,
            )
        logger.debug(initial_state1.variables)
        logger.debug(initial_state2.variables)

        for _ in range(search_strategy1.search_budget + search_strategy2.search_budget):
            draw = random.random()
            if draw < 0.5:
                state = search_strategy1.next_state()
                new_sample1 = sample1
                new_sample1.update_from_variables(state)
                new_sample2 = sample2
            else:
                state = search_strategy2.next_state()
                new_sample2 = sample2
                new_sample2.update_from_variables(state)
                new_sample1 = sample1
            # print(f"new samples{new_sample1.tokenized_code}")
            new_prediction = predictor(new_sample1, new_sample2)
            new_metric = metric(new_prediction)

            if draw < 0.5:
                search_strategy1.visit_state(state, new_metric)
            else:
                search_strategy2.visit_state(state, new_metric)

            if new_metric < original_metric:
                break

            if verbose:
                pbar.update(1)
        # print(search_strategy1._best_state,)
        return (
            search_strategy1._best_state,
            search_strategy2._best_state,
            min(
                initial_state1.score - search_strategy1._best_state.score,
                initial_state2.score - search_strategy2._best_state.score,
            ),
        )
