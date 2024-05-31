import ast
import json
import os
import random

import numpy as np
import torch
from loguru import logger


class Example(object):
    """A single training/test example."""

    def __init__(
        self,
        idx,
        code,
        source,
        target,
    ):
        self.idx = idx
        self.code = code
        self.source = source
        self.target = target


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


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(
        self,
        example_id,
        source_ids,
        target_ids,
        source_mask,
        target_mask,
    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask


def convert_examples_to_features(examples, tokenizer, args, stage=None):
    features = []
    for example_index, example in enumerate(examples):
        # source
        source_tokens = tokenizer.tokenize(example.source)[: args.max_source_length - 2]
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length

        # target
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[
                : args.max_target_length - 2
            ]
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length

        if example_index < 5:
            if stage == "train":
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))

                logger.info(
                    "source_tokens: {}".format(
                        [x.replace("\u0120", "_") for x in source_tokens]
                    )
                )
                logger.info("source_ids: {}".format(" ".join(map(str, source_ids))))
                logger.info("source_mask: {}".format(" ".join(map(str, source_mask))))

                logger.info(
                    "target_tokens: {}".format(
                        [x.replace("\u0120", "_") for x in target_tokens]
                    )
                )
                logger.info("target_ids: {}".format(" ".join(map(str, target_ids))))
                logger.info("target_mask: {}".format(" ".join(map(str, target_mask))))

        features.append(
            InputFeatures(
                example_index,
                source_ids,
                target_ids,
                source_mask,
                target_mask,
            )
        )
    return features


class VariableCollector(ast.NodeVisitor):
    def __init__(self):
        self.variables = set()
        self.defined_functions = set()

    def visit_FunctionDef(self, node):
        # Record function names to ignore them later
        self.defined_functions.add(node.name)
        # Visit function body for local variables
        self.generic_visit(node)

    def visit_Name(self, node):
        # Only add the name if it's not a function name
        if (
            isinstance(node.ctx, (ast.Load, ast.Store))
            # and node.id not in self.defined_functions
        ):
            self.variables.add(node.id)
        self.generic_visit(node)


def get_variable_locations(variables: list[str], source_tokens: list[str]) -> list[list[int]]:
    results = {variable: [] for variable in variables}

    for i, token in enumerate(source_tokens):
        if token in variables:
            results[token].append(i)

    return [results[variable] for variable in variables]