# Randomly mutate a given piece of code to attack the target code model
# Attack model: Swap two variables, randomly replace variable names from the candidates
# Search method: budgeted random search

import json
import keyword
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

modify_one_variable = False  # if True, only one variable is changed in total
indirect_attack = True  # if True, the attack is considered indirect

file_path = "data/eval_synthetic/eval__VARIABLE_MISUSE__SStuB.txt-00000-of-00300"
model_name = "microsoft/codebert-base"
model_path = "trained/codebert"

# Candidate file expecting a list of lists. In the nested list, the first element is the variable name and the second element is the score for that variable
candidate_file = "results/randomness_codebert_space.json"  # "results/randomness_codebert.json"  # "results/variables_map.json"

max_iterations = 60
seed = 42
log_frequency = 100

blacklist = set(keyword.kwlist)

blacklist.add("<unk>")
blacklist.add("<pad>")
blacklist.add("<mask>")


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
    new_tokens = list(tokens.copy())
    for i, variable in enumerate(base_variables):
        for location in variable_locs[variable]:
            new_tokens[location] = permutation_variables[i]
    return tuple(new_tokens)


def replace_variable(state_variables, candidates):
    state_variables = list(state_variables)
    idx = random.randint(0, len(state_variables) - 1)
    state_variables[idx] = random.choice(candidates)

    return tuple(state_variables)


def generate_adversarial(model, tokenizer, program, candidates):
    original_error_loc, _ = model_forward_pass(model, tokenizer, program)
    predicted_variable = program["source_tokens"][original_error_loc + 1]

    logger.debug(f"Ground truth location {program['error_location']}")
    logger.debug(f"Predicted location {original_error_loc}")

    variable_locations = extract_variables(program)

    if indirect_attack and predicted_variable in variable_locations:
        variable_locations.pop(predicted_variable)

    if len(variable_locations) == 0:
        return False, None

    base_variables = tuple(variable_locations.keys())
    state_variables = base_variables
    explored_set = {base_variables}
    original_tokens = program["source_tokens"]

    for i in range(max_iterations):
        new_state = replace_variable(state_variables, candidates)

        # Check if the state is already explored of the state is not valid (duplicated variables)
        if new_state in explored_set or len(new_state) != len(set(new_state)):
            continue

        explored_set.add(new_state)
        permuted_tokens = generate_new_tokens(
            original_tokens, variable_locations, base_variables, new_state
        )

        program["source_tokens"] = permuted_tokens
        new_error_loc, _ = model_forward_pass(model, tokenizer, program)

        if new_error_loc != original_error_loc:
            logger.debug(f"New location {new_error_loc}")
            return True, new_state

        # only change one variable name
        if modify_one_variable:
            state_variables = base_variables
        else:
            state_variables = new_state

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
    with open(candidate_file, "r", encoding="utf-8") as file:
        variables_map = json.load(file)
        candidates = [
            k[0]
            for k in variables_map["ranked_token"]
            if k[0] not in blacklist and k[0].isidentifier()
        ]
        candidates = [candidate.removeprefix("Ġ") for candidate in candidates]

    logger.info(f"Candidates: {candidates[:10]}")

    model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

    results = []

    direct_attack = 0

    for i in tqdm(range(0, 3000)):
        program = programs[i]

        if not program["has_bug"]:
            continue

        adversarial, permutation = generate_adversarial(
            model, tokenizer, program, candidates
        )
        results.append((i, adversarial, permutation))

        if i % log_frequency == 0:
            log_results(results)

    log_results(results)
