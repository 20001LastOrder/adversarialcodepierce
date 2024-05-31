import glob

import networkx as nx
import pandas as pd
from transformers import DataCollatorForTokenClassification


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


def tokenize_and_align_labels(examples, tokenizer, max_length=512):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, max_length=max_length
    )
    error = examples["error_location"]
    repair = examples["repair_locations"]
    h = examples["has_bug"]
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

def generate_data_token_classification(data):
    tokens = data["source_tokens"][1:]
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