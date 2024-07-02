import argparse
import json
import keyword
import os
import sys
from functools import partial

import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from acp.datasets.code_summarization import CodeGlueXSummarization
from acp.metrics.bleu import BLEU
from acp.samples.python import PythonSample
from acp.search import AdversarialSearch
from acp.strategies import (
    HeuristicSearch,
    PermutationSearch,
    RandomSearch,
    RandomVectorSearch,
)
from acp.strategies.base import Candidate

blacklist = set(keyword.kwlist)

blacklist.add("<unk>")
blacklist.add("<pad>")
blacklist.add("<mask>")

logger.remove(0)
logger.add(sys.stderr, level="INFO", serialize=False)


def main(args):
    dataset = CodeGlueXSummarization(filename=args.test_filename)
    sample = dataset[12][0]

    print(sample.tokenized_code)

    sample.mask_function_definitions()
    sample.mask_function_calls()
    sample.mask_variable_names()
    sample.remove_comments()

    print(sample.tokenized_code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_filename", type=str, required=True)

    args = parser.parse_args()
    main(args)
