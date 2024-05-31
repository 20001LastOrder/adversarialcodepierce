import json
import keyword
import random
import sys
from collections import defaultdict

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
ranked_file_name = "results/randomness_codebert_variable_subtoken.json"  # "results/randomness_codebert_space.json"  # "randomness_codebert_variables.json" #
max_iterations = 60
log_frequency = 100
seed = 42
blacklist = set(keyword.kwlist)

blacklist.add("<unk>")
blacklist.add("<pad>")
blacklist.add("<mask>")


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def entropy(logits):
    prob = torch.softmax(logits, dim=0)
    dist = Categorical(probs=prob)
    return dist.entropy()


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


def mask_variable(tokens, variable_locs, substitute_token):
    new_tokens = []

    for i, token in enumerate(tokens):
        if i in variable_locs:
            new_tokens.append(substitute_token)
        else:
            new_tokens.append(token)

    return new_tokens


def load_vocabulary(file_path):
    with open(file_path, "r") as file:
        vocabulary = json.load(file)
    vocabulary = list(vocabulary.items())

    return sorted(vocabulary, key=lambda x: x[1])


@torch.no_grad()
def model_forward_pass(model, tokenizer, program):
    program = generate_data_token_classification(program)
    tokenized = tokenize_and_align_labels(program, tokenizer)
    input_ids = torch.tensor(tokenized["input_ids"]).unsqueeze(0)

    # logger.error(tokenizer.convert_ids_to_tokens(tokenized["input_ids"]))

    attention_mask = torch.tensor(tokenized["attention_mask"]).unsqueeze(0)
    labels = torch.tensor(tokenized["labels"])

    outputs = model(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
    )
    localization_labels = labels[:, 0, 1:]
    localization_logits = outputs.logits[:, 1:, 0]
    localization_logits[localization_labels == -100] = float("-inf")

    localization_candidate_logits = localization_logits[localization_labels != -100]
    error_loc_pred = torch.argmax(localization_candidate_logits)

    return error_loc_pred, localization_candidate_logits


def get_token_location(input_ids, word_ids, variable_locs):
    locs = []
    for loc in variable_locs:
        locs.append(loc - 1)  # the #NEWLINE token at the beginning is removed
    variable_locs = set(locs)

    token_locs = []
    for i in range(len(input_ids)):
        if word_ids[i] in variable_locs:
            token_locs.append(i)

    return token_locs


def log_results(results):
    logger.info(f"Total records: {len(results)}")
    count = 0
    for i, adversarial, _ in results:
        if adversarial:
            count += 1

    logger.info(f"Adversarial found: {count / len(results) * 100:.2f}%")
    logger.info(f"Direct attack: {direct_attack / len(results) * 100:.2f}%")


def select_variable(variables, candidate, current_state):
    new_state = list(current_state)

    idx = random.randint(0, len(variables) - 1)
    new_state[idx] = candidate

    return tuple(new_state)


def generate_new_tokens(tokens, variable_locs, base_variables, permutation_variables):
    new_tokens = list(tokens.copy())
    for i, variable in enumerate(base_variables):
        for location in variable_locs[variable]:
            new_tokens[location] = permutation_variables[i]
    return tuple(new_tokens)


@torch.no_grad()
def generate_adversarial(model, tokenizer, program, candidates):
    variable_locations = extract_variables(program)
    error_loc_pred, _ = model_forward_pass(model, tokenizer, program)

    # consider the new line token at the beginning
    predicted_variable = program["source_tokens"][error_loc_pred + 1]

    original_tokens = program["source_tokens"]
    logger.debug(original_tokens[program["error_location"]])
    logger.debug(f"predicted_error_location: {error_loc_pred}")

    if indirect_attack and predicted_variable in variable_locations:
        variable_locations.pop(predicted_variable)

    if len(variable_locations) == 0:
        return False, None

    variables = tuple(variable_locations.keys())
    current_state = variables

    explored_set = {tuple(variables)}

    for i in range(max_iterations):
        candidate = candidates[i]

        # logger.error(tokenizer.convert_ids_to_tokens(candidate_tokens["input_ids"]))

        new_state = select_variable(variables, candidate, current_state)

        if new_state in explored_set or len(new_state) != len(set(new_state)):
            continue

        tokens = generate_new_tokens(
            original_tokens, variable_locations, variables, new_state
        )

        program["source_tokens"] = tokens
        error_loc_perturb, _ = model_forward_pass(model, tokenizer, program)

        if error_loc_perturb != error_loc_pred:
            return True, new_state

        # enable only changing one variable
        if modify_one_variable:
            current_state = variables
        else:
            current_state = new_state

    return False, None


if __name__ == "__main__":
    set_seed(seed)
    programs = []
    with open(file_path, "r") as file:
        for line in file:
            programs.append(json.loads(line))

    with open(ranked_file_name, "r", encoding="utf-8") as file:
        ranked_tokens = list(json.load(file)["ranked_token"])
        candidates = [ranked[0].removeprefix("Ġ") for ranked in ranked_tokens]
        candidates = [
            candidate
            for candidate in candidates
            if candidate.isidentifier() and candidate not in blacklist
        ]

    logger.info(f"Candidates: {candidates[:10]}")
    model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

    results = []

    direct_attack = 0

    for i in tqdm(range(3000)):
        program = programs[i]

        original_variable = program["source_tokens"][program["error_location"]]
        if not program["has_bug"]:
            continue

        adversarial, variable = generate_adversarial(
            model, tokenizer, program, candidates
        )
        if variable == original_variable:
            direct_attack += 1
        results.append((i, adversarial, variable))

        if i % log_frequency == 0:
            log_results(results)
        # break

    log_results(results)
