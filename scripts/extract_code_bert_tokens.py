"""
Extract tokens that are not changed when training RoBRETa to obtain the CodeBert.
"""

import torch
from transformers import AutoModel, AutoTokenizer


def euclidean_distance(a, b):
    return torch.nn.functional.pairwise_distance(a, b, eps=0)


def main():
    roberta_model = AutoModel.from_pretrained("FacebookAI/roberta-base")
    codebert_model = AutoModel.from_pretrained("microsoft/graphcodebert-base")

    roberta_tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")

    # print(roberta_tokenizer("Ukraine"))

    robert_word_embedding = roberta_model.embeddings.word_embeddings.weight
    codebert_word_embedding = codebert_model.embeddings.word_embeddings.weight

    tokens = roberta_tokenizer("Ukraine", return_tensors="pt")

    print(codebert_model.embeddings(input_ids=tokens["input_ids"]))
    print(roberta_model.embeddings(input_ids=tokens["input_ids"]))

    print(torch.isclose(robert_word_embedding, codebert_word_embedding))

    equal_rows = torch.isclose(robert_word_embedding, codebert_word_embedding).all(
        dim=1
    )

    print(equal_rows.shape)

    print(f"Number of equal rows: {equal_rows.sum()}")


if __name__ == "__main__":
    main()
