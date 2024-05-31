import argparse
import json


def main(config):
    with open(config.file_path, "r", encoding="utf-8") as file:
        tokens = json.load(file)["ranked_token"]

    filtered_tokens = []
    for token in tokens:
        if token[0].startswith("Ġ"):
            filtered_tokens.append(token)

    with open(config.output_file, "w", encoding="utf-8") as file:
        json.dump({"ranked_token": filtered_tokens}, file, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file_path",
        type=str,
        help="Path to the file containing the tokens",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to the output file",
    )

    args = parser.parse_args()

    main(args)
