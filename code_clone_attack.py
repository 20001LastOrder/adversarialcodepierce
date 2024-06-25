import argparse
import json
import keyword
import os
import sys
from functools import partial

import datasets
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from acp.datasets import BigCloneBenchDataset
from acp.metrics.em import EM
from acp.samples.java import JavaSample
from acp.search import AdversarialSearchPairs
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


def model_func(model, tokenizer, args, program1: JavaSample, program2: JavaSample):
    code1 = " ".join(program1.tokenized_code)
    code2 = " ".join(program2.tokenized_code)
    # print(program1.tokenized_code)
    # print(code2)

    tokenized = tokenizer(
        [[code1, code2]],
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
    )

    input_ids = torch.tensor(tokenized["input_ids"]).to(device)
    attention_mask = torch.tensor(tokenized["attention_mask"]).to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits

    return torch.argmax(outputs).item()


# def model_func_random_vec(
#     model,
#     tokenizer,
#     args,
#     search_strategy: RandomVectorSearch,
#     program: dict,
#     input_sample: PythonSample,
# ):
#     example = generate_data_token_classification(input_sample.tokenized_code, program)
#     tokenized = tokenize_and_align_labels(example, tokenizer, args.max_length)

#     embedder = model.get_input_embeddings()
#     new_input = search_strategy.get_new_input(
#         embedder, tokenized, input_sample.tokenized_code, device
#     ).unsqueeze(0)
#     attention_mask = torch.tensor(tokenized["attention_mask"]).unsqueeze(0).to(device)
#     labels = torch.tensor(tokenized["labels"])

#     outputs = model(
#         inputs_embeds=new_input,
#         attention_mask=attention_mask,
#     )

#     localization_labels = labels[:, 0, 1:]
#     localization_logits = outputs.logits[:, 1:, 0]
#     localization_logits[localization_labels == -100] = float("-inf")

#     localization_candidate_logits = localization_logits[localization_labels != -100]
#     error_loc_pred = torch.argmax(localization_candidate_logits)

#     return error_loc_pred.item()


def evaluation_metric(original, adversarial):
    metrics = EM()
    return metrics.evaluate(adversarial, original)


def log_metrics(attack_results):
    success = sum(attack_results)
    logger.info(f"Success rate: {success / len(attack_results) * 100:.2f}%")


def main(args):
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint).to(
        device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, add_prefix_space=True)
    hg_dataset = datasets.load_dataset(
        "antolin/bigclonebench_interduplication", split="test"
    )
    hg_dataset = hg_dataset.filter(lambda x: x["label"] == 1)
    # convert to a list of dictionaries
    examples = []
    for i in tqdm(range(3000)):
        examples.append(hg_dataset[i])

    dataset = BigCloneBenchDataset(examples=examples)
    search = AdversarialSearchPairs()

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
    gt_results = []

    for i in tqdm(range(500)):
        program1, program2, gt_result = dataset[i]
        strategy1 = strategy_cls(candidates=candidates, search_budget=30)
        strategy2 = strategy_cls(candidates=candidates, search_budget=30)
        original_prediction = model_func(model, tokenizer, args, program1, program2)

        if args.strategy == "random_vector":
            strategy = strategy_cls(
                mask_token=tokenizer.mask_token, candidates=candidates, search_budget=60
            )
            # model_output_func = partial(
            #     model_func_random_vec, model, tokenizer, args, strategy, program
            # )
        else:
            model_output_func = partial(model_func, model, tokenizer, args)

        adversarial_sample1, adversarial_sample2, _ = search.search(
            sample1=program1,
            sample2=program2,
            predictor=model_output_func,
            search_strategy1=strategy1,
            search_strategy2=strategy2,
            verbose=False,
            metric=partial(evaluation_metric, original_prediction),
        )

        program1.update_from_variables(adversarial_sample1)
        program2.update_from_variables(adversarial_sample2)

        adversarial_prediction = model_func(model, tokenizer, args, program1, program2)

        attack_results.append((adversarial_prediction != original_prediction))
        predicted_results.append(adversarial_prediction)

        original_results.append(original_prediction)
        gt_results.append(gt_result)

        if i % args.log_freq == 0:
            log_metrics(attack_results)

    log_metrics(attack_results)
    # print accuracy
    correct = 0
    for i in range(len(gt_results)):
        if gt_results[i] == original_results[i]:
            correct += 1
    print(f"Accuracy: {correct / len(gt_results) * 100:.2f}%")

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
    parser.add_argument("--adv_candidate_file", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--strategy", type=str, default="random")
    parser.add_argument("--log_freq", default=100, type=int)
    parser.add_argument("--output_folder", type=str)

    args = parser.parse_args()
    main(args)
