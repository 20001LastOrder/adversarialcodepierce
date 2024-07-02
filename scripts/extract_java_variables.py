import argparse
import json
from collections import defaultdict

import datasets
from tqdm import tqdm

from acp.samples.java import JavaSample


def main(args):
    dataset = datasets.load_dataset(args.dataset)
    # combine train, test and valid splits
    dataset = datasets.concatenate_datasets(
        [dataset["train"], dataset["test"], dataset["valid"]]
    )
    variables = defaultdict(int)

    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        sample = JavaSample(
            code=" ".join(item["tokens"]),
            tokenized_code=item["tokens"],
        )

        for variable in sample.variables:
            variables[variable.name] += 1

    with open("variables_map_java.json", "w") as f:
        json.dump(variables, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="antolin/tlc_interduplication")

    args = parser.parse_args()
    main(args)
