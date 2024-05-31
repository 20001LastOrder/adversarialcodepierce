from loguru import logger

from acp.samples.python import PythonSample

sample = {
    "repo": "soimort/you-get",
    "path": "src/you_get/extractors/miomio.py",
    "func_name": "sina_xml_to_url_list",
    "original_string": 'def sina_xml_to_url_list(xml_data):\n    """str->list\n    Convert XML to URL List.\n    From Biligrab.\n    """\n    rawurl = []\n    dom = parseString(xml_data)\n    for node in dom.getElementsByTagName(\'durl\'):\n        url = node.getElementsByTagName(\'url\')[0]\n        rawurl.append(url.childNodes[0].data)\n    return rawurl',
    "language": "python",
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
    "docstring": "str->list\n    Convert XML to URL List.\n    From Biligrab.",
    "docstring_tokens": [
        "str",
        "-",
        ">",
        "list",
        "Convert",
        "XML",
        "to",
        "URL",
        "List",
        ".",
        "From",
        "Biligrab",
        ".",
    ],
    "sha": "b746ac01c9f39de94cac2d56f665285b0523b974",
    "url": "https://github.com/soimort/you-get/blob/b746ac01c9f39de94cac2d56f665285b0523b974/src/you_get/extractors/miomio.py#L41-L51",
    "partition": "test",
}


def test_python_sample_code():
    python_sample = PythonSample(
        code=sample["code"], tokenized_code=sample["code_tokens"]
    )
    tokens = sample["code_tokens"]

    gt_variables = set(["dom", "rawurl", "url", "xml_data", "node"])

    variables = python_sample.variables
    assert len(variables) == len(gt_variables)

    for variable in variables:
        assert variable.name in gt_variables

        for location in variable.locations:
            assert tokens[location] == variable.name

        gt_variables.remove(variable.name)
