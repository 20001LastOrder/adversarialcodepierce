# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import

import argparse
import ast
import json
import keyword
import os
import random
import sys
from io import open
from itertools import cycle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import (AutoTokenizer, RobertaConfig, RobertaModel,
                          RobertaTokenizer)
from transformers.tokenization_utils_base import CharSpan

from random_search_code_summarization.bleu import (bleu, nltk_sentence_bleu,
                                                   splitPuncts)
from random_search_code_summarization.model import Seq2Seq
from random_search_code_summarization.utils import (
    VariableCollector, convert_examples_to_features, get_variable_locations,
    read_examples)

blacklist = set(keyword.kwlist)

blacklist.add("<unk>")
blacklist.add("<pad>")
blacklist.add("<mask>")


MODEL_CLASSES = {"roberta": (RobertaConfig, RobertaModel, RobertaTokenizer)}


logger.remove(0)
logger.add(sys.stderr, level="INFO", serialize=False)


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def log_results(original_bleu_scores, adversarial_bleu_scores):
    logger.info(f"Total records: {len(original_bleu_scores)}")

    original_bleu_score = sum(original_bleu_scores) / len(original_bleu_scores)
    adversarial_bleu_score = sum(adversarial_bleu_scores) / len(adversarial_bleu_scores)

    logger.info(f"Original BLEU score: {original_bleu_score}")
    logger.info(f"Adversarial BLEU score: {adversarial_bleu_score}")
    logger.info(f"drop: {original_bleu_score - adversarial_bleu_score}")


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type: e.g. roberta",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model: e.g. roberta-base",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--load_model_path",
        default=None,
        type=str,
        help="Path to trained model: Should contain the .bin files",
    )
    ## Other parameters
    parser.add_argument(
        "--test_filename",
        default=None,
        type=str,
        help="The test filename. Should contain the .jsonl files for this task.",
    )

    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--max_source_length",
        default=64,
        type=int,
        help="The maximum total source sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_target_length",
        default=32,
        type=int,
        help="The maximum total target sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )

    parser.add_argument(
        "--beam_size", default=10, type=int, help="beam size for beam search"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )

    # Adversarial args
    parser.add_argument(
        "--adv_mode",
        type=str,
        choices=["random", "heuristic", "permutation", "random_vector"],
        default="random",
    )

    parser.add_argument("--output_folder", type=str)

    parser.add_argument("--log_freq", type=int, default=20)

    parser.add_argument("--adv_candidate_file", type=str)

    parser.add_argument("--adv_max_iterations", type=int, default=60)
    # print arguments
    args = parser.parse_args()
    logger.info(args)

    with open(args.adv_candidate_file, "r", encoding="utf-8") as f:
        candidates = json.load(f)["ranked_token"]
        candidates = [candidate[0].removeprefix("Ġ") for candidate in candidates]
        candidates = [k for k in candidates if k not in blacklist and k.isidentifier()]

    logger.info(f"Sample candidates: {candidates[:10]}")

    # Setup CUDA, GPU & distributed training
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    # Set seed
    set_seed(args.seed)
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
    )

    # build model
    encoder = model_class.from_pretrained(args.model_name_or_path, config=config)
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=config.hidden_size, nhead=config.num_attention_heads
    )
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        config=config,
        beam_size=args.beam_size,
        max_length=args.max_target_length,
        sos_id=tokenizer.cls_token_id,
        eos_id=tokenizer.sep_token_id,
    )
    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(device)

    model.eval()

    logger.info("Test file: {}".format(args.test_filename))
    eval_examples = read_examples(args.test_filename)

    original_bleu_scores = []
    adversarial_bleu_scores = []

    for i, eval_example in enumerate(tqdm(eval_examples[:200])):
        if args.adv_mode == "random_vector":
            lowest_bleu, bleu_score = generate_adversarial_random_vector(
                model, tokenizer, eval_example, args, device
            )
        else:
            lowest_bleu, bleu_score = generate_adversarial(
                model, tokenizer, eval_example, candidates, args, device
            )

        # if lowest_bleu == bleu_score:
        #     logger.error(i)
        #     break

        original_bleu_scores.append(bleu_score)
        adversarial_bleu_scores.append(lowest_bleu)

        if i % args.log_freq == 0:
            log_results(original_bleu_scores, adversarial_bleu_scores)

    output_file_name = os.path.join(args.output_folder, f"{args.adv_mode}.json")
    log_results(original_bleu_scores, adversarial_bleu_scores)

    logger.info(f"BLEU score: {sum(original_bleu_scores) / len(original_bleu_scores)}")
    logger.info(
        f"Lowest BLEU score: {sum(adversarial_bleu_scores) / len(adversarial_bleu_scores)}"
    )

    adversarial_success = 0
    for original, adversarial in zip(original_bleu_scores, adversarial_bleu_scores):
        if original > adversarial:
            adversarial_success += 1

    logger.info(
        f"Adversarial success rate: {adversarial_success / len(original_bleu_scores)}"
    )

    results = []

    for original, adversarial in zip(original_bleu_scores, adversarial_bleu_scores):
        results.append({"original bleu": original, "adversarial bleu": adversarial})

    results = pd.DataFrame(results)
    results.to_csv(output_file_name, index=False)


@torch.no_grad()
def model_forward(model, tokenizer, example, args, device):
    eval_features = convert_examples_to_features(
        [example], tokenizer, args, stage="test"
    )

    all_source_ids = torch.tensor(
        [f.source_ids for f in eval_features], dtype=torch.long
    )

    all_source_mask = torch.tensor(
        [f.source_mask for f in eval_features], dtype=torch.long
    )

    all_source_ids = all_source_ids.to(device)
    all_source_mask = all_source_mask.to(device)

    preds = model(source_ids=all_source_ids, source_mask=all_source_mask)
    t = preds[0][0].cpu().numpy()
    t = list(t)
    if 0 in t:
        t = t[: t.index(0)]
    text = tokenizer.decode(t, clean_up_tokenization_spaces=False)

    return text


def select_next_variable(variables, current_state, candidate):
    variable_idx = random.randint(0, len(variables) - 1)

    current_state[variable_idx] = candidate

    return current_state


def replace_variable(source_tokens, variable_locations, current_state):
    for idx, replacement in enumerate(current_state):
        for location in variable_locations[idx]:
            source_tokens[location] = replacement

    return " ".join(source_tokens)


def generate_adversarial(model, tokenizer, example, candidates, args, device):
    collector = VariableCollector()
    collector.visit(ast.parse(example.code))
    variables = list(collector.variables)
    # also add the function definition names
    # variables = list(collector.variables.symmetric_difference(collector.defined_functions))

    # also consider the function definition names
    variable_locations = get_variable_locations(variables, example.source)
    original_tokens = example.source

    example.source = " ".join(example.source)
    original_prediction = model_forward(model, tokenizer, example, args, device)
    # original_bleu = bleu(
    #     [splitPuncts(example.target)], splitPuncts(original_prediction)
    # )[0]
    original_bleu = nltk_sentence_bleu(example.target, original_prediction)

    current_state = variables.copy()

    visited_states = {tuple(current_state)}

    lowest_bleu = original_bleu

    for i in range(args.adv_max_iterations):
        if args.adv_mode == "random":
            candidate = random.choice(candidates)
            current_state = select_next_variable(variables, current_state, candidate)
        elif args.adv_mode == "heuristic":
            candidate = candidates[i]
            current_state = select_next_variable(variables, current_state, candidate)
        elif args.adv_mode == "permutation":
            current_state = np.random.permutation(variables).tolist()
        # logger.error(current_state)
        if tuple(current_state) in visited_states:
            continue

        new_tokens = replace_variable(
            original_tokens, variable_locations, current_state
        )

        example.source = new_tokens
        prediction = model_forward(model, tokenizer, example, args, device)
        # logger.error(prediction)
        # bleu_score = bleu([splitPuncts(example.target)], splitPuncts(prediction))[0]
        bleu_score = nltk_sentence_bleu(example.target, prediction)

        lowest_bleu = min(bleu_score, lowest_bleu)

        visited_states.add(tuple(current_state))
        # current_state = variables.copy()

        if original_bleu > lowest_bleu:
            break

    return lowest_bleu, original_bleu


def mask_variable(tokens, variable_locs, substitute_token):
    new_tokens = []

    for i, token in enumerate(tokens):
        if i in variable_locs:
            new_tokens.append(substitute_token)
        else:
            new_tokens.append(token)

    return new_tokens


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


def get_token_location(word_ids, variable_locs):
    locs = []

    for loc in variable_locs:
        locs.append(loc)
    variable_locs = set(locs)

    token_locs = []
    for i in range(len(word_ids)):
        if word_ids[i] in variable_locs:
            token_locs.append(i)

    return token_locs


@torch.no_grad()
def model_forward_embedding(model, source_embedding, source_mask, tokenizer):

    preds = model(source_embedding=source_embedding, source_mask=source_mask)
    t = preds[0][0].cpu().numpy()
    t = list(t)
    if 0 in t:
        t = t[: t.index(0)]
    text = tokenizer.decode(t, clean_up_tokenization_spaces=False)

    return text


def contains(span1, span2):
    return span1.start <= span2.start and span1.end >= span2.end


def remap_word_ids(word_ids, tokenized_input, original_tokens):
    token_spans = []
    start = 0
    for token in original_tokens:
        token_spans.append(CharSpan(start=start, end=start + len(token)))
        start += len(token) + 1  # account for space

    new_word_ids = []

    for word_id in word_ids:
        if word_id is None:
            new_word_ids.append(None)
            continue

        span = tokenized_input.word_to_chars(word_id)
        for j, token_span in enumerate(token_spans):
            if contains(token_span, span):
                new_word_ids.append(j)
                break
    return new_word_ids


def generate_adversarial_random_vector(model, tokenizer, example, args, device):
    collector = VariableCollector()
    collector.visit(ast.parse(example.code))
    variables = list(collector.variables)
    variable_locations = get_variable_locations(variables, example.source)
    variable_locations = {
        variable: variable_locations[i] for i, variable in enumerate(variables)
    }
    original_tokens = example.source

    example.source = " ".join(original_tokens)
    original_prediction = model_forward(model, tokenizer, example, args, device)
    original_bleu = bleu(
        [splitPuncts(example.target)], splitPuncts(original_prediction)
    )[0]

    # original_bleu = nltk_sentence_bleu(example.target, original_prediction)

    # logger.error(original_tokens)
    # logger.error(f"original prediction: {original_prediction}")
    # logger.error(f"original bleu: {original_bleu}")
    # logger.error(f"ground truth: {example.target}")

    current_state = {}
    for i in range(args.adv_max_iterations):
        masked_tokens, current_state = generate_next_state(
            original_tokens,
            current_state,
            variables,
            variable_locations,
            tokenizer,
            model.config.hidden_size,
        )
        # logger.error(current_state.keys())
        # logger.error(masked_tokens)
        logger.debug(f"length of masked tokens: {len(masked_tokens)}")

        new_tokens = " ".join(masked_tokens)
        example.source = new_tokens

        eval_features = convert_examples_to_features(
            [example], tokenizer, args, stage="test"
        )

        source_ids = torch.tensor(eval_features[0].source_ids, dtype=torch.long).to(
            device
        )
        source_masks = torch.tensor(eval_features[0].source_mask, dtype=torch.long).to(
            device
        )

        tokenized_input = tokenizer(
            new_tokens, max_length=args.max_source_length, truncation=True
        )

        word_ids = tokenized_input.word_ids(0)
        new_word_ids = remap_word_ids(word_ids, tokenized_input, masked_tokens)

        logger.debug(f"new word ids: {new_word_ids}")

        # replace embedding
        embeddings = model.encoder.embeddings.word_embeddings(source_ids)
        for variable, random_vector in current_state.items():
            token_locations = get_token_location(
                new_word_ids, variable_locations[variable]
            )
            logger.debug(f"variable locations: {variable_locations[variable]}")
            logger.debug(f"token_locations: {token_locations}")

            random_vector = random_vector.to(device)
            embeddings[token_locations, :] = random_vector

        prediction = model_forward_embedding(
            model, embeddings.unsqueeze(0), source_masks.unsqueeze(0), tokenizer
        )
        # logger.error(prediction)

        bleu_score = bleu([splitPuncts(example.target)], splitPuncts(prediction))[0]
        # bleu_score = nltk_sentence_bleu(example.target, prediction)
        # logger.error(splitPuncts(prediction))

        if bleu_score < original_bleu:
            break

        # update the masked tokens for the next iteration
        original_tokens = masked_tokens

    return bleu_score, original_bleu


if __name__ == "__main__":
    main()
