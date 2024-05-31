from loguru import logger

from acp.samples.python import PythonSample
from acp.strategies.base import Candidate
from acp.strategies.heuristic import HeuristicSearch

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


def test_heuristic_search():
    python_sample = PythonSample(
        code=sample["code"], tokenized_code=sample["code_tokens"]
    )
    variable_names = [variable.name for variable in python_sample.variables]
    variable_locations = [variable.locations for variable in python_sample.variables]

    candidate_names = ["var1", "var2", "var3"]
    candidate_scores = [0.1, 0, 0.2]

    candidates = [
        Candidate(content=name, score=score)
        for name, score in zip(candidate_names, candidate_scores)
    ]

    heuristic_search = HeuristicSearch(candidates=candidates)

    initial_state = heuristic_search.initiate_state(python_sample, 0.5)

    assert initial_state.score == 0.5
    assert initial_state.variables == variable_names
    assert initial_state.locations == variable_locations

    next_state = heuristic_search.next_state()

    assert next_state.variables != variable_names
    assert (
        sum(
            [
                new_variable != old_variable
                for new_variable, old_variable in zip(
                    next_state.variables, variable_names
                )
            ]
        )
        == 1
    )

    assert "var3" in next_state.variables
    for variable in next_state.variables:
        assert variable in candidate_names or variable in variable_names

    heuristic_search.visit_state(next_state, 0.1)

    assert heuristic_search._current_state == next_state
    assert heuristic_search._best_state == next_state
