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

from preprocess import generate_data_token_classification, tokenize_and_align_labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.remove(0)
logger.add(sys.stderr, level="INFO", serialize=False)

modify_one_variable = False  # if True, only one variable is changed in total
indirect_attack = True

file_path = "data/eval_synthetic/eval__VARIABLE_MISUSE__SStuB.txt-00000-of-00300"
program_idx = 2
model_name = "microsoft/codebert-base"
model_path = "trained/codebert"
max_iterations = 60
log_frequency = 100
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


def generate_next_state(
    tokens, current_state, variables, variable_locations, tokenizer, embedding_size
):
    selected_variable = np.random.choice(variables)
    new_state = current_state.copy()
    masked_tokens = mask_variable(
        tokens, set(variable_locations[selected_variable]), tokenizer.mask_token
    )
    random_embeddings = torch.nn.Embedding(1, embedding_size)
    torch.nn.init.normal_(random_embeddings.weight, mean=0, std=0.02)
    random_mask_embedding = random_embeddings.weight
    new_state[selected_variable] = random_mask_embedding

    return masked_tokens, new_state


@torch.no_grad()
def generate_adversarial(model, tokenizer, program):
    variable_locations = extract_variables(program)

    error_loc_pred, localization_candidate_logits = model_forward_pass(
        model, tokenizer, program
    )

    predicted_variable = program["source_tokens"][error_loc_pred + 1]

    original_tokens = program["source_tokens"]
    logger.debug(original_tokens[program["error_location"]])
    logger.debug(f"predicted_error_location: {error_loc_pred}")

    if indirect_attack and predicted_variable in variable_locations:
        variable_locations.pop(predicted_variable)

    if len(variable_locations) == 0:
        return False, None

    variables = list(variable_locations.keys())
    current_state = {}

    # Replace the variable names one by one with a random vector
    for i in range(max_iterations):
        masked_tokens, current_state = generate_next_state(
            original_tokens,
            current_state,
            variables,
            variable_locations,
            tokenizer,
            model.roberta.embeddings.word_embeddings.weight.size()[1],
        )
        program["source_tokens"] = masked_tokens

        program_data = generate_data_token_classification(program)
        tokenized = tokenize_and_align_labels(program_data, tokenizer)

        # replace embeddings
        embeddings = model.roberta.embeddings.word_embeddings(
            torch.tensor(tokenized["input_ids"]).to(device)
        )

        for variable, random_vector in current_state.items():
            token_locations = get_token_location(
                tokenized["input_ids"],
                tokenized["word_ids"][0],
                variable_locations[variable],
            )

            logger.debug(f"variable locations: {variable_locations[variable]}")
            logger.debug(f"token_locations: {token_locations}")

            random_vector = random_vector.to(device)
            embeddings[token_locations, :] = random_vector.clone()

        error_loc_perturb, localization_candidate_logits = model_forward_embeds(
            model, embeddings, tokenized["attention_mask"], tokenized["labels"]
        )

        logger.debug(f"predicted_error_location: {error_loc_perturb}")

        if error_loc_perturb != error_loc_pred:
            return True, variable

        # enable only changing one variable
        if modify_one_variable:
            current_state = {}
        else:
            original_tokens = masked_tokens

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

    for i in tqdm(range(3000)):
        program = programs[i]

        original_variable = program["source_tokens"][program["error_location"]]
        if not program["has_bug"]:
            continue

        adversarial, variable = generate_adversarial(model, tokenizer, program)
        if variable == original_variable:
            direct_attack += 1
        results.append((i, adversarial, variable))

        if i % log_frequency == 0:
            log_results(results)

    log_results(results)
