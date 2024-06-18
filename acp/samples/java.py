from collections import defaultdict
from functools import cached_property
from typing import Optional

from pydantic import computed_field
from tree_sitter_languages import get_language, get_parser

from acp.samples.base import BaseSample, Variable

parser = get_parser("java")

edge_case_types = {"method_invocation", "field_access", "method_declaration"}


def collect_varaible_tokens(node, pos_to_token_map, variables=[]):
    if node.type == "identifier" and (
        node.parent.type not in edge_case_types
        or node
        == node.parent.children[
            0
        ]  # check if the identifier is the first child of the parent
    ):
        start_byte = node.start_byte
        end_byte = node.end_byte
        token_idx = pos_to_token_map[(start_byte, end_byte)]
        variables.append(token_idx)
    for child in node.children:
        collect_varaible_tokens(child, pos_to_token_map, variables)

    return variables


def get_pos_to_token_map(code_tokens, connector=" "):
    pos_to_token_map = {}
    pos = 0
    for i, token in enumerate(code_tokens):
        start = pos
        end = pos + len(token)
        pos_to_token_map[(start, end)] = i
        pos = end + len(connector)

    return pos_to_token_map


class JavaSample(BaseSample):
    @computed_field
    @cached_property
    def ast(self) -> Optional[object]:
        try:
            return parser.parse(bytes(self.code, "utf-8"))
        except Exception:
            return None

    def create_variables(self) -> list[Variable]:
        pos_to_token_map = get_pos_to_token_map(self.tokenized_code)
        variables = []
        collect_varaible_tokens(self.ast.root_node, pos_to_token_map, variables)
        variable_map = defaultdict(list)

        for variable_idx in variables:
            variable_map[self.tokenized_code[variable_idx]].append(variable_idx)

        return [
            Variable(
                name=variable, locations=variable_map[variable]
            )
            for variable in variable_map
        ]

    def tokenize(self, code: str) -> list[str]:
        return code.split(" ")
