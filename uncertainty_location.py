import json
from collections import defaultdict

import torch
from torch.distributions import Categorical
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

from preprocess import (generate_data_token_classification,
                        tokenize_and_align_labels)

file_path = "data/eval_synthetic/eval__VARIABLE_MISUSE__SStuB.txt-00000-of-00300"
program_idx = 4
model_name = "microsoft/codebert-base"
model_path = "trained/codebert"
max_iterations = 60

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
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    localization_labels = labels[:, 0, 1:]
    localization_logits = outputs.logits[:, 1:, 0]
    localization_logits[localization_labels == -100] = float("-inf")
    # error_loc_pred = torch.argmax(localization_logits, dim=1) + 2 # the CLS token and the space token added at the beginning

    localization_candidate_logits = localization_logits[localization_labels != -100]
    error_loc_pred = torch.argmax(localization_candidate_logits) 

    return error_loc_pred, localization_candidate_logits



if __name__ == "__main__":
    programs = []
    with open(file_path, "r") as file:
        for line in file:
            programs.append(json.loads(line))

    program = programs[program_idx]

    if not program["has_bug"]:
        print("Program does not have a bug")
        exit()


    variables = extract_variables(program)

    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

    error_loc_pred, localization_candidate_logits = model_forward_pass(model, tokenizer, program)
    # error_loc_entropy = entropy(localization_candidate_logits)\
    print(localization_candidate_logits.shape)
    error_loc_entropy = torch.nn.functional.cross_entropy(localization_candidate_logits, error_loc_pred)
    
    print(f"largest entropy: {entropy(torch.ones_like(localization_candidate_logits))}")
    print(f"current entropy: {error_loc_entropy}")

    original_tokens = program["source_tokens"]
    print(original_tokens[program["error_location"]])
    print(f"predicted_error_location: {error_loc_pred}")
    vocabulary = load_vocabulary("results/variable_distances_codebert.json")

    variable_ranking = []

    for variable, locs in tqdm(variables.items()):
        masked_tokens = mask_variable(original_tokens, set(locs), tokenizer.mask_token)
        program["source_tokens"] = masked_tokens
        error_loc_perturb, logits_perturb = model_forward_pass(model, tokenizer, program)
        # error_loc_entropy_perturb = entropy(logits_perturb)
        error_loc_entropy_perturb = torch.nn.functional.cross_entropy(logits_perturb, error_loc_pred)

        variable_ranking.append((variable, error_loc_entropy_perturb))

    variable_ranking.sort(key=lambda x: x[1], reverse=True)
    
    used_variables = set()
    substitute_map = {}
    current_worst = error_loc_entropy
    adversarial_found = False

    program["source_tokens"] = original_tokens
    for variable, score in tqdm(variable_ranking):
        print("Substituting variable: ", variable)
        current_substitute = variable
        for i in tqdm(range(max_iterations)):
            if vocabulary[i][0] in used_variables:
                continue

            substitute_token = vocabulary[i][0]
            program["source_tokens"] = mask_variable(program["source_tokens"], set(variables[variable]), substitute_token)
            error_loc_perturb, logits_perturb = model_forward_pass(model, tokenizer, program)

            loss = torch.nn.functional.cross_entropy(logits_perturb, error_loc_pred)
            if loss > current_worst:
                current_worst = loss
                current_substitute = substitute_token

            if error_loc_perturb != error_loc_pred:
                print("Adversarial example found!")
                print("error_loc_perturb: ", error_loc_perturb)
                print("subsitute token: ", current_substitute)
                adversarial_found = True
                break

        print(f"Current worst: {current_worst}")
        program["source_tokens"] = mask_variable(program["source_tokens"], set(variables[variable]), current_substitute)
        used_variables.add(current_substitute)
        substitute_map[variable] = current_substitute

        if adversarial_found:
            print(program["source_tokens"] )
            break

    print("Substitute map: ", substitute_map)
        

