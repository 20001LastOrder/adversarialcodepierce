import argparse
import json

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def average_embedding(embeddings, attention_mask=None):
    if attention_mask is None:
        attention_mask = torch.ones(embeddings.shape[:-1], device=embeddings.device)
    return torch.sum(embeddings * attention_mask.unsqueeze(-1), dim=1) / torch.clamp(
        attention_mask.sum(dim=1, keepdim=True), min=1e-9
    )


def get_embedding(texts, embedder, tokenizer):
    tokenized = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    embeddings = embedder(tokenized["input_ids"])

    return average_embedding(embeddings, tokenized["attention_mask"])


def euclidean_distance(a, b):
    return torch.nn.functional.pairwise_distance(a, b, eps=0)


def get_variable_distances(
    variables_map, pretrained_embedding, finetuned_embedding, tokenizer, batch_size
):
    variables = list(variables_map.keys())
    variable_distances = dict()

    for i in tqdm(range(0, len(variables), batch_size)):
        batch = variables[i : i + batch_size]
        embeddings = get_embedding(batch, pretrained_embedding, tokenizer)
        finetuned_embeddings = get_embedding(batch, finetuned_embedding, tokenizer)
        distances = (
            euclidean_distance(embeddings, finetuned_embeddings).cpu().detach().tolist()
        )

        for j, variable in enumerate(batch):
            variable_distances[variable] = distances[j]

    return variable_distances


def main(args):
    pretrained_model = AutoModel.from_pretrained(args.model_name)
    pretrained_embedding = pretrained_model.embeddings.to(device)

    fine_tuned_model = AutoModel.from_pretrained(args.model_path)
    fine_tuned_embedding = fine_tuned_model.embeddings.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    with open(args.variables_map_path, "r") as file:
        variables_map = json.load(file)

    distances = get_variable_distances(
        variables_map, pretrained_embedding, fine_tuned_embedding, tokenizer, args.batch_size
    )

    with open("variable_distances.json", "w") as file:
        json.dump(distances, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="microsoft/codebert-base")
    parser.add_argument("--variables_map_path", type=str, default="variables_map.json")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    main(args)
