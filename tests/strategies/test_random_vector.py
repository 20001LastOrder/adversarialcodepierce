import torch
from loguru import logger
from transformers import AutoTokenizer

from acp.samples.python import PythonSample
from acp.strategies.base import Candidate
from acp.strategies.random_vector import RandomVectorSearch

sample = sample = {
    "code": 'def sina_xml_to_url_list(xml_data):\n    """str->list\n    Convert XML to URL List.\n    From Biligrab.\n    """\n    rawurl = []\n    dom = parseString(xml_data)\n    for node in dom.getElementsByTagName(\'durl\'):\n        url = node.getElementsByTagName(\'url\')[0]\n        rawurl.append(url.childNodes[0].data)\n    return rawurl',
    "code_tokens": [
        "def",
        "sina_xml_to_url_list",
        "(",
        "xml_data",
        ")",
        ":",
        "rawurl",
        "=",
        "[",
        "]",
        "dom",
        "=",
        "parseString",
        "(",
        "xml_data",
        ")",
        "for",
        "node",
        "in",
        "dom",
        ".",
        "getElementsByTagName",
        "(",
        "'durl'",
        ")",
        ":",
        "url",
        "=",
        "node",
        ".",
        "getElementsByTagName",
        "(",
        "'url'",
        ")",
        "[",
        "0",
        "]",
        "rawurl",
        ".",
        "append",
        "(",
        "url",
        ".",
        "childNodes",
        "[",
        "0",
        "]",
        ".",
        "data",
        ")",
        "return",
        "rawurl",
    ],
}


def test_random_search():
    python_sample = PythonSample(
        code=sample["code"], tokenized_code=sample["code_tokens"]
    )
    variable_names = [variable.name for variable in python_sample.variables]
    variable_locations = [variable.locations for variable in python_sample.variables]
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    mask_token = tokenizer.bos_token

    random_search = RandomVectorSearch(mask_token=tokenizer.bos_token)

    initial_state = random_search.initiate_state(python_sample, 0.5)

    assert initial_state.score == 0.5
    assert initial_state.variables == variable_names
    assert initial_state.locations == variable_locations

    next_state = random_search.next_state()

    assert next_state.variables != variable_names
    assert mask_token in next_state.variables

    embedder = torch.nn.Embedding(tokenizer.vocab_size, 768)
    python_sample.update_from_variables(next_state)

    new_input = random_search.get_new_input(
        embedder=embedder,
        tokenizer=tokenizer,
        sample=python_sample,
        max_length=256,
        device="cpu",
        validate=True,
    )

    assert new_input.shape[1] == 768

    logger.info(new_input.shape)


def test_random_search_roberta():
    python_sample = PythonSample(
        code=sample["code"], tokenized_code=sample["code_tokens"]
    )
    variable_names = [variable.name for variable in python_sample.variables]
    variable_locations = [variable.locations for variable in python_sample.variables]
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
    mask_token = tokenizer.mask_token

    random_search = RandomVectorSearch(mask_token=tokenizer.mask_token)

    initial_state = random_search.initiate_state(python_sample, 0.5)

    assert initial_state.score == 0.5
    assert initial_state.variables == variable_names
    assert initial_state.locations == variable_locations

    next_state = random_search.next_state()

    assert next_state.variables != variable_names
    assert mask_token in next_state.variables

    embedder = torch.nn.Embedding(tokenizer.vocab_size, 768)
    python_sample.update_from_variables(next_state)

    new_input = random_search.get_new_input(
        embedder=embedder,
        tokenizer=tokenizer,
        sample=python_sample,
        max_length=256,
        device="cpu",
        validate=True,
    )

    assert new_input.shape[1] == 768

    logger.info(new_input.shape)
