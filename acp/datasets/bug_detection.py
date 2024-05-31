import json
from collections import defaultdict
from typing import Tuple

from pydantic import model_validator

from acp.datasets.base import BaseDataset
from acp.samples.base import Variable
from acp.samples.python import PythonSample


class Python150BugDetection(BaseDataset):
    filename: str
    examples: list[dict] = None

    @model_validator(mode="after")
    def read_examples(self):
        self.examples = []
        with open(self.filename, "r") as file:
            for line in file:
                example = json.loads(line)
                if example["has_bug"]:
                    self.examples.append(example)

    @property
    def size(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[PythonSample, dict]:
        example = self.examples[idx]
        # skip the first new line token
        tokens = example["source_tokens"][1:]
        code = " ".join(tokens)
        variables = extract_variables(
            tokens, example["repair_candidates"], example["error_location"]
        )

        return (
            PythonSample(code=code, tokenized_code=tokens, variables=variables),
            example,
        )


def extract_variables(
    tokens: list[str], repair_candidates: list, error_location: int
) -> list[Variable]:
    # locations -1 because of the newline token is removed

    variable_locations = set()
    for candidate in repair_candidates:
        if type(candidate) is int:
            variable_locations.add(candidate - 1)

    variable_locations.add(error_location - 1)

    variables = defaultdict(list)
    for loc in variable_locations:
        variable = tokens[loc]
        variables[variable].append(loc)

    variables = [
        Variable(name=variable, locations=locations)
        for variable, locations in variables.items()
    ]

    return variables
