import json
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


file_path = "data/eval_synthetic/eval__VARIABLE_MISUSE__SStuB.txt-00000-of-00300"
model_name = "microsoft/codebert-base"
model_path = "trained/codebert"
ranked_file_name = "results/randomness_codebert.json"
max_iterations = 60
seed = 42


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


@torch.no_grad()
def model_forward_embeds(model, inputs_embeds, attention_mask, labels):
    inputs_embeds = torch.tensor(inputs_embeds).unsqueeze(0)
    attention_mask = torch.tensor(attention_mask).unsqueeze(0)

    labels = torch.tensor(labels)

    outputs = model(
        inputs_embeds=inputs_embeds.to(device),
        attention_mask=attention_mask.to(device),
    )
    localization_labels = labels[:, 0, 1:]
    localization_logits = outputs.logits[:, 1:, 0]
    localization_logits[localization_labels == -100] = float("-inf")
    # error_loc_pred = torch.argmax(localization_logits, dim=1) + 2 # the CLS token and the space token added at the beginning

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


def generate_adversarial(model, tokenizer, program):
    with open(ranked_file_name, "r") as file:
        ranked_tokens = json.load(file)["ranked_token"]

    ranked_tokens = ranked_tokens[:max_iterations]

    variable_locations = extract_variables(program)

    error_loc_pred, localization_candidate_logits = model_forward_pass(
        model, tokenizer, program
    )

    error_loc_entropy = torch.nn.functional.cross_entropy(
        localization_candidate_logits, error_loc_pred
    )

    logger.debug(
        f"largest entropy: {entropy(torch.ones_like(localization_candidate_logits))}"
    )
    logger.debug(f"current entropy: {error_loc_entropy}")

    original_tokens = program["source_tokens"]
    logger.debug(original_tokens[program["error_location"]])
    logger.debug(f"predicted_error_location: {error_loc_pred}")

    variable_ranking = []

    for variable, locs in variable_locations.items():
        masked_tokens = mask_variable(original_tokens, set(locs), tokenizer.mask_token)
        program["source_tokens"] = masked_tokens
        error_loc_perturb, logits_perturb = model_forward_pass(
            model, tokenizer, program
        )
        # error_loc_entropy_perturb = entropy(logits_perturb)
        error_loc_entropy_perturb = torch.nn.functional.cross_entropy(
            logits_perturb, error_loc_pred
        )

        variable_ranking.append((variable, error_loc_entropy_perturb))

    variable_ranking.sort(key=lambda x: x[1], reverse=True)
    used_variables = set()
    best_entropy = error_loc_entropy
    for variable, score in variable_ranking:
        best_token = ""
        for token, _ in ranked_tokens[3:]:
            if token in used_variables:
                continue
            program["source_tokens"] = mask_variable(
                original_tokens, set(variable_locations[variable]), token
            )

            error_loc_perturb, logits_perturb = model_forward_pass(
                model, tokenizer, program
            )
            error_loc_entropy = torch.nn.functional.cross_entropy(
                logits_perturb, error_loc_perturb
            )

            logger.debug(f"current entropy: {error_loc_entropy}")
            logger.debug(f"predicted_error_location: {error_loc_perturb}")

            if error_loc_perturb != error_loc_pred:
                logger.debug(program["source_tokens"])
                return True, variable

            if error_loc_entropy > best_entropy:
                best_entropy = error_loc_entropy
                best_token = token
                break
        if best_token != "":
            original_tokens = mask_variable(
                original_tokens, set(variable_locations[variable]), best_token
            )
            used_variables.add(best_token)

    return False, None


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

    for i in tqdm(range(3000)):
        program = programs[i]

        original_variable = program["source_tokens"][program["error_location"]]
        if not program["has_bug"]:
            continue

        adversarial, variable = generate_adversarial(model, tokenizer, program)
        if variable == original_variable:
            direct_attack += 1
        results.append((i, adversarial, variable))

    logger.info(f"Total records: {len(results)}")
    count = 0
    for i, adversarial, variable in results:
        if adversarial:
            count += 1

    logger.info(f"Adversarial found: {count / len(results) * 100:.2f}%")
    logger.info(f"Direct attack: {direct_attack / len(results) * 100:.2f}%")
