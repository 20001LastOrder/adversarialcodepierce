import json
from typing import Any, Tuple

from pydantic import BaseModel, model_validator

from acp.datasets.base import BaseDataset
from acp.samples.python import PythonSample


class Example(BaseModel):
    """A single training/test example."""

    idx: int
    code: str
    source: list[str]
    target: str


def read_examples(filename):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if "idx" not in js:
                js["idx"] = idx
            code = " ".join(js["code_tokens"]).replace("\n", " ")
            code_tokens = code.strip().split()
            nl = " ".join(js["docstring_tokens"]).replace("\n", "")
            nl = " ".join(nl.strip().split())
            examples.append(
                Example(
                    idx=idx,
                    code=js["code"],
                    source=code_tokens,
                    target=nl,
                )
            )
    return examples


class CodeGlueXSummarization(BaseDataset):
    filename: str
    examples: list[Example] = None

    @model_validator(mode="after")
    def read_examples(self):
        self.examples = read_examples(self.filename)

    @property
    def size(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[PythonSample, Any]:
        example = self.examples[idx]
        return (
            PythonSample(code=example.code, tokenized_code=example.source),
            example.target,
        )
