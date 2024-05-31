# Randomly mutate a given piece of code to attack the target code model
# Attack model: Swap two variables, randomly replace variable names from the candidates
# Search method: budgeted random search

import json
import random
import sys
from collections import defaultdict

import numpy as np
import torch
from loguru import logger
from torch.distributions import Categorical
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

from preprocess import (generate_data_token_classification,
                        tokenize_and_align_labels)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.remove(0)
logger.add(sys.stderr, level="INFO", serialize=False)

indirect_attack = True  # if True, the attack is considered indirect

file_path = "data/eval_synthetic/eval__VARIABLE_MISUSE__SStuB.txt-00000-of-00300"
model_name = "microsoft/codebert-base"
model_path = "trained/codebert"
max_iterations = 60
seed = 42
log_frequency = 100


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def extract_variables(program):
    locations = set()
    tokens = program["source_tokens"]
    for loc in program["repair_candidates"]:
        if type(loc) == int:
            locations.add(loc)

    locations.add(program["error_location"])

    variables = defaultdict(list)
    for loc in locations:
        variable = tokens[loc]
        variables[variable].append(loc)

    return variables


@torch.no_grad()
def model_forward_pass(model, tokenizer, program):
    program = generate_data_token_classification(program)
    tokenized = tokenize_and_align_labels(program, tokenizer)
    input_ids = torch.tensor(tokenized["input_ids"]).unsqueeze(0)
    attention_mask = torch.tensor(tokenized["attention_mask"]).unsqueeze(0)
    labels = torch.tensor(tokenized["labels"])

    outputs = model(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
    )
    localization_labels = labels[:, 0, 1:]
    localization_logits = outputs.logits[:, 1:, 0]
    localization_logits[localization_labels == -100] = float("-inf")
    # error_loc_pred = torch.argmax(localization_logits, dim=1) + 2 # the CLS token and the space token added at the beginning

    localization_candidate_logits = localization_logits[localization_labels != -100]
    error_loc_pred = torch.argmax(localization_candidate_logits)

    return error_loc_pred, localization_candidate_logits


def generate_new_tokens(tokens, variable_locs, base_variables, permutation_variables):
    new_tokens = tokens.copy()
    for i, variable in enumerate(base_variables):
        for location in variable_locs[variable]:
            new_tokens[location] = permutation_variables[i]
    return new_tokens


def generate_adversarial(model, tokenizer, program):
    original_error_loc, _ = model_forward_pass(model, tokenizer, program)

    logger.debug(f"Ground truth location {program['error_location']}")
    logger.debug(f"Predicted location {original_error_loc}")

    # consider the new line token at the beginning
    predicted_variable = program["source_tokens"][original_error_loc + 1]

    variable_locations = extract_variables(program)
    if indirect_attack and predicted_variable in variable_locations:
        variable_locations.pop(predicted_variable)

    base_variables = tuple(variable_locations.keys())
    explored_set = {base_variables}
    original_tokens = program["source_tokens"]

    for i in range(max_iterations):
        permutation = tuple(np.random.permutation(base_variables))
        if permutation in explored_set:
            continue

        explored_set.add(permutation)

        permuted_tokens = generate_new_tokens(
            original_tokens, variable_locations, base_variables, permutation
        )

        program["source_tokens"] = permuted_tokens
        new_error_loc, _ = model_forward_pass(model, tokenizer, program)

        if new_error_loc != original_error_loc:
            logger.debug(f"New location {new_error_loc}")
            return True, permutation

    return False, None


def log_results(results):
    logger.info(f"Total records: {len(results)}")
    count = 0
    for i, adversarial, _ in results:
        if adversarial:
            count += 1

    logger.info(f"Adversarial found: {count / len(results) * 100:.2f}%")
    logger.info(f"Direct attack: {direct_attack / len(results) * 100:.2f}%")


if __name__ == "__main__":
    set_seed(seed)
    programs = []
    with open(file_path, "r") as file:
        for line in file:
            programs.append(json.loads(line))

    model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

    results = []

    direct_attack = 0

    for i in tqdm(range(1, 3000)):
        program = programs[i]

        if not program["has_bug"]:
            continue

        adversarial, permutation = generate_adversarial(model, tokenizer, program)
        results.append((i, adversarial, permutation))

        if i % log_frequency == 0:
            log_results(results)

    log_results(results)
