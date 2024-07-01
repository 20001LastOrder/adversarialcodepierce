import argparse
import json
import keyword
import os
import sys
from functools import partial

import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

from acp.datasets import Python150BugDetection
from acp.metrics.em import EM
from acp.samples.python import PythonSample
from acp.search import AdversarialSearch
from acp.strategies import (
    HeuristicSearch,
    PermutationSearch,
    RandomSearch,
    RandomVectorSearch,
)
from acp.strategies.base import Candidate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
blacklist = set(keyword.kwlist)

blacklist.add("<unk>")
blacklist.add("<pad>")
blacklist.add("<mask>")
logger.remove(0)
logger.add(sys.stderr, level="INFO", serialize=False)


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = -100  # labels[word_id]
            new_labels.append(label)
    return new_labels


def tokenize_and_align_labels(example, tokenizer, max_length=512):
    tokenized_inputs = tokenizer(
        example["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
    )
    error = example["error_location"]
    repair = example["repair_locations"]
    h = example["has_bug"]
    all_labels = []
    word_ids = tokenized_inputs.word_ids(0)
    alignment = align_labels_with_tokens(error, word_ids)
    alignment[0] = 1 if h else 0
    alignment_repair = align_labels_with_tokens(repair, word_ids)
    all_labels.append([alignment, alignment_repair])

    tokenized_inputs["labels"] = all_labels
    tokenized_inputs["word_ids"] = [
        tokenized_inputs.word_ids(i) for i in range(len(all_labels))
    ]
    return tokenized_inputs


def generate_data_token_classification(tokens, data):
    has_bug = data["has_bug"]
    error_location = [0] * len(tokens)
    if has_bug:
        error_location[data["error_location"] - 1] = 1
    repair_locations = [-100] * len(tokens)
    for j in data["repair_candidates"]:
        repair_locations[j - 1] = 0
    if has_bug:
        for j in data["repair_targets"]:
            repair_locations[j - 1] = 1
    return {
        "tokens": tokens,
        "has_bug": has_bug,
        "error_location": error_location,
        "repair_locations": repair_locations,
    }


def model_func(model, tokenizer, args, program: dict, input: PythonSample):
    # input.mask_variable_names()
    # input.mask_function_calls(use_regex=True)
    # input.mask_function_calls(use_regex=True)
    example = generate_data_token_classification(input.tokenized_code, program)
    tokenized = tokenize_and_align_labels(example, tokenizer, args.max_length)

    input_ids = torch.tensor(tokenized["input_ids"]).unsqueeze(0).to(device)
    attention_mask = torch.tensor(tokenized["attention_mask"]).unsqueeze(0).to(device)
    labels = torch.tensor(tokenized["labels"])

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    localization_labels = labels[:, 0, 1:]
    localization_logits = outputs.logits[:, 1:, 0]
    localization_logits[localization_labels == -100] = float("-inf")

    localization_candidate_logits = localization_logits[localization_labels != -100]
    error_loc_pred = torch.argmax(localization_candidate_logits)

    return error_loc_pred.item()


def model_func_random_vec(
    model,
    tokenizer,
    args,
    search_strategy: RandomVectorSearch,
    program: dict,
    input_sample: PythonSample,
):
    example = generate_data_token_classification(input_sample.tokenized_code, program)
    tokenized = tokenize_and_align_labels(example, tokenizer, args.max_length)

    embedder = model.get_input_embeddings()
    new_input = search_strategy.get_new_input(
        embedder, tokenized, input_sample.tokenized_code, device
    ).unsqueeze(0)
    attention_mask = torch.tensor(tokenized["attention_mask"]).unsqueeze(0).to(device)
    labels = torch.tensor(tokenized["labels"])

    outputs = model(
        inputs_embeds=new_input,
        attention_mask=attention_mask,
    )

    localization_labels = labels[:, 0, 1:]
    localization_logits = outputs.logits[:, 1:, 0]
    localization_logits[localization_labels == -100] = float("-inf")

    localization_candidate_logits = localization_logits[localization_labels != -100]
    error_loc_pred = torch.argmax(localization_candidate_logits)

    return error_loc_pred.item()


def evaluation_metric(original, adversarial):
    metrics = EM()
    return metrics.evaluate(adversarial, original)


def log_metrics(attack_results):
    success = sum(attack_results)
    logger.info(f"Success rate: {success / len(attack_results) * 100:.2f}%")


def main(args):
    model = AutoModelForTokenClassification.from_pretrained(args.checkpoint).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, add_prefix_space=True)
    dataset = Python150BugDetection(filename=args.test_filename)
    search = AdversarialSearch()

    with open(args.adv_candidate_file, "r", encoding="utf-8") as f:
        candidates = json.load(f)["ranked_token"]
        candidates = [
            (candidate[0].removeprefix("Ġ"), -candidate[1]) for candidate in candidates
        ]
        candidates = [
            k for k in candidates if k[0] not in blacklist and k[0].isidentifier()
        ]

    candidates = [Candidate(content=k[0], score=k[1]) for k in candidates]

    strategy_cls = None
    if args.strategy == "random":
        strategy_cls = RandomSearch
    elif args.strategy == "heuristic":
        strategy_cls = HeuristicSearch
    elif args.strategy == "permutation":
        strategy_cls = PermutationSearch
    elif args.strategy == "random_vector":
        strategy_cls = RandomVectorSearch

    attack_results = []
    original_results = []
    predicted_results = []
    labels = []

    for i in tqdm(range(len(dataset))):
        dataset_item, program = dataset[i]
        # strategy = strategy_cls(candidates=candidates, search_budget=60)
        original_prediction = model_func(model, tokenizer, args, program, dataset_item)

        labels.append(program["error_location"] - 1)  # exclude the [NEW_LINE] token

        if args.strategy == "random_vector":
            strategy = strategy_cls(
                mask_token=tokenizer.mask_token, candidates=candidates, search_budget=60
            )
            model_output_func = partial(
                model_func_random_vec, model, tokenizer, args, strategy, program
            )
        else:
            model_output_func = partial(model_func, model, tokenizer, args, program)

        adversarial_example, _ = search.search(
            dataset_item,
            predictor=model_output_func,
            search_strategy=strategy,
            verbose=False,
            metric=partial(evaluation_metric, original_prediction),
        )

        dataset_item.update_from_variables(adversarial_example)
        adversarial_prediction = model_func(
            model, tokenizer, args, program, dataset_item
        )

        attack_results.append((adversarial_prediction != original_prediction))
        original_results.append(original_prediction)
        predicted_results.append(adversarial_prediction)

        if i % args.log_freq == 0:
            log_metrics(attack_results)

    log_metrics(attack_results)

    # calculate accuracy
    equal = sum([1 for i, j in zip(original_results, labels) if i == j])
    logger.info(f"Accuracy: {equal / len(labels) * 100:.2f}%")

    output_name = f"{args.strategy}_{args.adv_candidate_file.split('/')[-1]}.csv"
    output_path = os.path.join(args.output_folder, output_name)

    df = pd.DataFrame(
        {
            "original": original_results,
            "adversarial": predicted_results,
            "is adversarial": attack_results,
        }
    )

    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--test_filename", type=str, required=True)
    parser.add_argument("--adv_candidate_file", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--strategy", type=str, default="random")
    parser.add_argument("--log_freq", default=100)
    parser.add_argument("--output_folder", type=str)

    args = parser.parse_args()
    main(args)
