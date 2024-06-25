from acp.datasets.base import BaseDataset
from acp.samples.java import JavaSample


class BigCloneBenchDataset(BaseDataset):
    examples: list[dict]

    @property
    def size(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        example = self.examples[idx]
        return (
            JavaSample(
                code=" ".join(example["tokens1"]),
                tokenized_code=example["tokens1"],
            ),
            JavaSample(
                code=" ".join(example["tokens2"]),
                tokenized_code=example["tokens2"],
            ),
            example["label"],
        )
