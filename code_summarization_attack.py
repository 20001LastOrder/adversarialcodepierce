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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            code = " ".join(code.strip().split())
            nl = " ".join(js["docstring_tokens"]).replace("\n", "")
            nl = " ".join(nl.strip().split())
            examples.append(
                {
                    "idx": idx,
                    "source": code,
                    "target": nl,
                }
            )
    return examples


def model_func(model, tokenizer, args, input: PythonSample):
    code = " ".join(input.tokenized_code)
    # print(code)
    model_inputs = tokenizer(
        code,
        max_length=args.max_source_length,
        padding="max_length",
        truncation=True,
    )

    input_ids = torch.tensor(model_inputs["input_ids"]).unsqueeze(0).to(device)
    attention_mask = (
        torch.tensor(model_inputs["attention_mask"]).unsqueeze(0).to(device)
    )

    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=args.max_target_length,
        num_beams=args.num_beams,
    )

    logger.debug(tokenizer.decode(generated[0], skip_special_tokens=True))
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def model_func_random_vec(
    model,
    tokenizer,
    args,
    search_strategy: RandomVectorSearch,
    input_sample: PythonSample,
):
    code = " ".join(input_sample.tokenized_code)
    tokenized_input = tokenizer(
        code,
        max_length=args.max_source_length,
        # padding="max_length",
        truncation=True,
    )
    embedder = model.get_input_embeddings()
    new_input = search_strategy.get_new_input(
        embedder, tokenized_input, input_sample.tokenized_code, device
    ).unsqueeze(0)
    attention_mask = (
        torch.tensor(tokenized_input["attention_mask"]).unsqueeze(0).to(device)
    )

    generated = model.generate(
        inputs_embeds=new_input,
        attention_mask=attention_mask,
        max_length=args.max_target_length,
        num_beams=args.num_beams,
    )

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def evaluate_metric(ground_truth, prediction):
    metrics = BLEU()
    return metrics.evaluate(prediction, ground_truth)


def log_metrics(original_bleus, adversarial_bleus):
    original_bleu = sum(original_bleus) / len(original_bleus)
    adversarial_bleu = sum(adversarial_bleus) / len(adversarial_bleus)
    logger.info(f"Original BLEU: {original_bleu}")
    logger.info(f"Adversarial BLEU: {adversarial_bleu}")

    success_rate = sum(
        1
        for original, adversarial in zip(original_bleus, adversarial_bleus)
        if original > adversarial
    ) / len(original_bleus)

    logger.info(f"Success rate: {success_rate}")


def main(args):
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    dataset = CodeGlueXSummarization(filename=args.test_filename)
    with open(args.adv_candidate_file, "r", encoding="utf-8") as f:
        candidates = json.load(f)["ranked_token"]
        candidates = [
            (candidate[0].removeprefix("Ġ"), -candidate[1]) for candidate in candidates
        ]
        candidates = [
            k for k in candidates if k[0] not in blacklist and k[0].isidentifier()
        ]

    candidates = [Candidate(content=k[0], score=k[1]) for k in candidates]
    logger.info(candidates[:10])
    search = AdversarialSearch()

    model_output_function = partial(model_func, model, tokenizer, args)

    strategy_cls = None
    if args.strategy == "random":
        strategy_cls = RandomSearch
    elif args.strategy == "heuristic":
        strategy_cls = HeuristicSearch
    elif args.strategy == "permutation":
        strategy_cls = PermutationSearch
    elif args.strategy == "random_vector":
        strategy_cls = RandomVectorSearch

    original_bleus = []
    adversarial_bleus = []
    original_predictions = []
    adversarial_predictions = []
    for i in tqdm(range(200)):
        dataset_item, target = dataset[i]

        strategy = strategy_cls(candidates=candidates, search_budget=60)
        if args.strategy == "random_vector":
            strategy = strategy_cls(mask_token=tokenizer.bos_token, candidates=candidates, search_budget=60)
            model_output_function = partial(
                model_func_random_vec, model, tokenizer, args, strategy
            )

        adversarial_example, _ = search.search(
            dataset_item,
            predictor=model_output_function,
            search_strategy=strategy,
            metric=partial(evaluate_metric, target),
            verbose=False,
        )

        original_bleus.append(strategy._initial_state.score)
        adversarial_bleus.append(adversarial_example.score)

        original_predictions.append(model_output_function(dataset_item))
        dataset_item.update_from_variables(adversarial_example)
        adversarial_predictions.append(model_output_function(dataset_item))

        if i % args.log_freq == 0:
            log_metrics(original_bleus, adversarial_bleus)

    log_metrics(original_bleus, adversarial_bleus)

    output_name = f"{args.strategy}_{args.adv_candidate_file.split('/')[-1]}.csv"
    output_path = os.path.join(args.output_folder, output_name)

    df = pd.DataFrame(
        {
            "original": original_predictions,
            "adversarial": adversarial_predictions,
            "original bleu": original_bleus,
            "adversarial bleu": adversarial_bleus,
        }
    )

    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--test_filename", type=str, required=True)
    parser.add_argument("--adv_candidate_file", type=str, required=True)
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--strategy", type=str, default="random")
    parser.add_argument("--num_beams", type=int, default=10)
    parser.add_argument("--log_freq", default=20)
    parser.add_argument("--output_folder", type=str)

    args = parser.parse_args()
    main(args)
