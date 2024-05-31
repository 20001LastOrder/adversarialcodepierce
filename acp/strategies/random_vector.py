import numpy as np
import torch
from loguru import logger
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import CharSpan

from acp.samples.base import BaseSample
from acp.strategies.base import BaseSearchStrategy
from acp.strategies.states import RandomVectorState


class RandomVectorSearch(BaseSearchStrategy):
    """
    Replace the embedding of a vector by a random initialized vector
    """

    random_seed: int = 42
    vector_size: int = 768
    mean: float = 0.0
    std: float = 0.02
    mask_token: str = "<mask>"

    def __init__(self, **kwargs):
        kwargs["candidates"] = []
        super().__init__(**kwargs)

        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)

    def get_random_vector(self) -> torch.Tensor:
        return torch.normal(mean=self.mean, std=self.std, size=(self.vector_size,))

    def initiate_state(
        self, sample: BaseSample, original_metric: float
    ) -> RandomVectorState:
        variables = sample.variables

        variable_names = [variable.name for variable in variables]
        variable_locations = [variable.locations for variable in variables]

        self._initial_state = RandomVectorState(
            variables=variable_names,
            locations=variable_locations,
            score=original_metric,
        )
        self._best_state = self._initial_state
        self._current_state = self._initial_state

        return self._initial_state

    def next_state(self) -> RandomVectorState:
        self._current_iteration += 1

        state = self._current_state.model_copy(deep=True)

        if len(state.variables) == 0:
            return state

        next_variable_location = np.random.choice(range(len(state.locations)))

        # set the location to the mask token
        state.variables[next_variable_location] = self.mask_token
        state.vectors[next_variable_location] = self.get_random_vector()
        self._current_state = state

        return state

    def get_new_input(
        self,
        embedder: torch.nn.Embedding,
        tokenized_input: torch.Tensor,
        code_tokens: list[str],
        device: str = "cpu",
        validate: bool = False,
        tokenizer: PreTrainedTokenizer = None,
    ) -> torch.Tensor:
        word_ids = tokenized_input.word_ids(0)

        new_word_ids = remap_word_ids(word_ids, tokenized_input, code_tokens)

        state = self._current_state

        input_ids = torch.tensor(tokenized_input["input_ids"]).to(device)

        embeddings = embedder(input_ids)

        if state is None:
            # return the original embedding if the state is None
            return embeddings

        for _, locations, vector in zip(
            state.variables, state.locations, state.vectors
        ):
            token_locations = get_token_location(new_word_ids, locations)

            if vector is None or len(vector) == 0:
                continue

            vector = vector.to(device)

            if validate:
                for location in token_locations:
                    assert (
                        tokenizer.convert_ids_to_tokens([input_ids[location]])[0]
                        == self.mask_token
                    )

            embeddings[token_locations, :] = vector

        return embeddings

    def visit_state(self, state: RandomVectorState, new_metric: float):
        state.score = new_metric

        if state.score < self._best_state.score:
            self._best_state = state

        self._current_state = state


def contains(span1, span2):
    return span1.start <= span2.start and span1.end >= span2.end


def remap_word_ids(word_ids, tokenized_input, original_tokens):
    token_spans = []
    start = 0
    for token in original_tokens:
        token_spans.append(CharSpan(start=start - 1, end=start + len(token) + 1))
        start += len(token) + 1  # account for space
    logger.debug(f"Token spans: {token_spans}")
    new_word_ids = []

    for word_id in word_ids:
        if word_id is None:
            new_word_ids.append(None)
            continue

        span = tokenized_input.word_to_chars(word_id)
        logger.debug(f"{span} {word_id}")
        for j, token_span in enumerate(token_spans):
            if contains(token_span, span):
                new_word_ids.append(j)
                break
    logger.debug(f"New word ids {new_word_ids}")
    return new_word_ids


def get_token_location(word_ids, variable_locs):
    locs = []

    for loc in variable_locs:
        locs.append(loc)
    variable_locs = set(locs)

    token_locs = []
    for i in range(len(word_ids)):
        if word_ids[i] in variable_locs:
            token_locs.append(i)

    return token_locs
