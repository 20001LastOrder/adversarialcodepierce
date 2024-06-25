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
        if (start_byte, end_byte) in pos_to_token_map:
            token_idx = pos_to_token_map[(start_byte, end_byte)]
            variables.append(token_idx)
    for child in node.children:
        collect_varaible_tokens(child, pos_to_token_map, variables)

    return variables


def get_function_definitions(node):
    function_definitions = []
    if node.type == "method_declaration":
        function_definitions.append(node)
    for child in node.children:
        function_definitions.extend(get_function_definitions(child))
    return function_definitions


def get_function_calls(node):
    function_calls = []
    if node.type == "method_invocation":
        function_calls.append(node)
    for child in node.children:
        function_calls.extend(get_function_calls(child))
    return function_calls


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
            Variable(name=variable, locations=variable_map[variable])
            for variable in variable_map
        ]

    def tokenize(self, code: str) -> list[str]:
        return code.split(" ")

    def collect_function_definition_ids(self):
        function_definitions = get_function_definitions(self.ast.root_node)
        pos_to_token_map = get_pos_to_token_map(self.tokenized_code)
        ids = []

        for function_definition in function_definitions:
            function_definition_identifier = None
            for child in function_definition.children:
                if child.type == "identifier":
                    function_definition_identifier = child
                    break

            start_byte = function_definition_identifier.start_byte
            end_byte = function_definition_identifier.end_byte

            if start_byte == end_byte:
                continue

            token_idx = pos_to_token_map[(start_byte, end_byte)]
            ids.append(token_idx)
        return ids

    def collect_variable_ids(self):
        ids = []
        for variable in self.variables:
            for location in variable.locations:
                ids.append(location)
        return ids

    def collect_function_call_ids(self):
        ids = []
        function_calls = get_function_calls(self.ast.root_node)
        pos_to_token_map = get_pos_to_token_map(self.tokenized_code)

        for function_call in function_calls:
            arglist_idx = -1
            for i, child in enumerate(function_call.children):
                if child.type == "argument_list":
                    arglist_idx = i
                    break
            function_call_identifier = function_call.children[arglist_idx - 1]
            start_byte = function_call_identifier.start_byte
            end_byte = function_call_identifier.end_byte

            if (start_byte, end_byte) not in pos_to_token_map:
                continue
            token_idx = pos_to_token_map[(start_byte, end_byte)]
            ids.append(token_idx)

        return ids

    def mask_function_definitions(self, mask_token: str = "<mask>") -> str:
        function_definition_ids = self.collect_function_definition_ids()

        for function_definition_id in function_definition_ids:
            self.tokenized_code[function_definition_id] = mask_token

        self.code = " ".join(self.tokenized_code)

    def mask_variables(self, mask_token: str = "<mask>") -> str:
        variable_ids = self.collect_variable_ids()

        for variable_id in variable_ids:
            self.tokenized_code[variable_id] = mask_token

        self.code = " ".join(self.tokenized_code)

    def mask_function_calls(self, mask_token: str = "<mask>") -> str:
        function_call_ids = self.collect_function_call_ids()

        for function_call_id in function_call_ids:
            self.tokenized_code[function_call_id] = mask_token

        self.code = " ".join(self.tokenized_code)

    def mask_all(self, mask_token: str = "<mask>") -> str:
        ids = []
        ids.extend(self.collect_function_definition_ids())
        ids.extend(self.collect_variable_ids())
        ids.extend(self.collect_function_call_ids())

        for id in ids:
            self.tokenized_code[id] = mask_token

        self.code = " ".join(self.tokenized_code)
