import json
import random
import sys

import numpy as np
import torch
from loguru import logger
from scipy.stats import kstest, norm
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

logger.remove(0)
logger.add(sys.stderr, level="INFO", serialize=False)

seed = 42
output_name = "randomness_codebert_variables_codeclonde.json"
candidate_file_name = "results/variables_map_java.json"


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def measure_distance_candidates(model, tokenizer, distribution, candidate_file_name):
    with open(candidate_file_name, "r", encoding="utf-8") as f:
        candidates = list(json.load(f).keys())

    ks_scores = []
    p_values = []
    variances = []

    for candidate in tqdm(candidates):
        tokens = tokenizer(candidate, return_tensors="pt")

        # remove the special tokens
        tokens["input_ids"] = tokens["input_ids"][0][1:-1]

        embeddings = (
            model.embeddings.word_embeddings(tokens["input_ids"]).detach().numpy()
        )

        # embeddings = model.encoder.embed_tokens(tokens["input_ids"]).detach().numpy()

        current_ks_scores = []
        current_p_values = []
        current_variances = []

        for embedding in embeddings:
            result = kstest(embedding, distribution.cdf)
            current_ks_scores.append(result.statistic)
            current_p_values.append(result.pvalue)
            current_variances.append(np.var(embedding))

        ks_scores.append(np.min(current_ks_scores))
        p_values.append(np.min(current_p_values))
        variances.append(np.max(current_variances))

    mean_score_model = np.mean(ks_scores)
    logger.info(f"Mean Variance score: {mean_score_model}")

    ranked_candidates = np.argsort(ks_scores).tolist()
    ranked_candidates = [candidates[i] for i in ranked_candidates]
    ks_scores = np.sort(
        ks_scores,
    ).tolist()

    results = [
        (ranked_candidates[i], ks_scores[i]) for i in range(len(ranked_candidates))
    ]

    return results


def measure_distance_tokens(model, tokenizer, distribution):
    embedding = model.embeddings.word_embeddings.weight.detach().numpy()
    # embedding = model.encoder.embed_tokens.weight.detach().numpy()
    ks_scores = []
    p_values = []
    variances = []

    for i in tqdm(range(embedding.shape[0])):
        result = kstest(embedding[i], distribution.cdf)
        ks_scores.append(result.statistic)
        p_values.append(result.pvalue)
        variances.append(np.var(embedding[i]))

    # mean_score_model = np.mean(ks_scores)
    mean_ks_distance = np.mean(ks_scores)
    logger.info(f"Mean Randomness score: {mean_ks_distance}")

    ranked_tokens = np.argsort(ks_scores).tolist()
    ranked_tokens = [tokenizer.convert_ids_to_tokens([i])[0] for i in ranked_tokens]
    ks_scores = np.sort(ks_scores).tolist()

    results = [(ranked_tokens[i], ks_scores[i]) for i in range(len(ranked_tokens))]
    return results


def main():
    model = AutoModel.from_pretrained("trained/code_clone/best_checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/codebert-base", add_prefix_space=True
    )

    distribution = norm(0, 0.02)

    random_embedding = torch.nn.Embedding(1000, 768)
    torch.nn.init.normal_(random_embedding.weight, mean=0, std=0.02)
    ks_scores = []
    p_values = []
    variances = []

    for i in tqdm(range(200)):
        random_sample = random_embedding(torch.tensor([i]))
        random_sample = random_sample.detach().numpy().flatten()
        result = kstest(random_sample, distribution.cdf)
        ks_scores.append(result.statistic)
        p_values.append(result.pvalue)
        variances.append(np.var(random_sample))

    mean_score = np.mean(ks_scores)
    logger.info(f"Mean randomness score: {mean_score}")

    if candidate_file_name is None or candidate_file_name == "":
        results = measure_distance_tokens(model, tokenizer, distribution)
    else:
        results = measure_distance_candidates(
            model, tokenizer, distribution, candidate_file_name
        )

    with open(output_name, "w", encoding="utf-8") as f:
        json.dump({"ranked_token": results}, f, ensure_ascii=False)


if __name__ == "__main__":
    set_seed(seed)
    main()
