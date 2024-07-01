import ast
import keyword
import re
import tokenize
from functools import cached_property
from io import BytesIO
from typing import Optional

from loguru import logger
from pydantic import computed_field

from acp.samples.base import BaseSample, Variable

__all__ = ["PythonSample"]


class VariableCollector(ast.NodeVisitor):
    def __init__(self):
        self.variables = set()
        self.defined_functions = set()
        self.function_calls = set()

    def visit_FunctionDef(self, node):
        # Record function names to ignore them later
        logger.debug(f"Function name {node.name}")
        self.defined_functions.add(node.name)
        # self.variables.add(node.name)
        for arg in node.args.args:
            self.variables.add(arg.arg)
        # Visit function body for local variables
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.function_calls.add(node.func.id)
        self.generic_visit(node)

    def visit_Name(self, node):
        # Only add the name if it's not a function name
        if isinstance(node.ctx, (ast.Store)) and node.id not in self.defined_functions:
            self.variables.add(node.id)
        self.generic_visit(node)


class PythonSample(BaseSample):
    @computed_field
    @cached_property
    def ast(self) -> Optional[object]:
        try:
            return ast.parse(self.code)
        except Exception:
            if self.variables is None:
                logger.error(f"Error parsing code {self.code}")
            return None

    def create_variables(self) -> list[Variable]:
        collector = VariableCollector()
        collector.visit(self.ast)

        variable_names = list(collector.variables)

        variable_locations = get_variable_locations(variable_names, self.tokenized_code)

        return [
            Variable(name=variable_names[i], locations=variable_locations[i])
            for i in range(len(variable_names))
        ]

    def mask_function_definitions(
        self, mask_token: str = "<mask>", use_regex=False
    ) -> str:
        if not use_regex:
            collector = VariableCollector()
            collector.visit(self.ast)

            function_locs = get_func_def_locations(
                collector.defined_functions, self.tokenized_code
            )

            for locs in function_locs:
                for loc in locs:
                    self.tokenized_code[loc] = mask_token
        else:
            pattern = re.compile(r"def (.+)\(")
            for i, token in enumerate(self.tokenized_code):
                if pattern.match(token):
                    self.tokenized_code[i] = f"def {mask_token}("

    def mask_function_calls(self, func_prefix="function", use_regex=False) -> str:
        if not use_regex:
            collector = VariableCollector()
            collector.visit(self.ast)

            function_calls = list(collector.function_calls)
            function_call_locations = get_func_def_locations(
                function_calls, self.tokenized_code
            )

            for i, locs in enumerate(function_call_locations):
                for loc in locs:
                    self.tokenized_code[loc] = f"{func_prefix}_{i}"
        else:
            func_name_map = {}
            func_id = 0
            for i, token in enumerate(self.tokenized_code):
                if token == "(" and self.tokenized_code[i - 1] not in keyword.kwlist:
                    func_name = self.tokenized_code[i - 1]
                    if func_name not in func_name_map:
                        func_name_map[func_name] = f"{func_prefix}_{func_id}"
                        func_id += 1
                    self.tokenized_code[i - 1] = func_name_map[func_name]

    def tokenize(self, code: str) -> list[str]:
        tokens = tokenize.tokenize(BytesIO(code.encode("utf-8")).readline)
        return [token.string for token in tokens if token.type != tokenize.ENCODING]

    def remove_comments(self):
        lines = self.code.split("\n")
        comment_lengths = []
        for line in lines:
            if "#" in line:
                line = line.split("#")[1]
                comment_lengths.append(len(line.split(" ")))
        new_tokens = []
        comment_idx = 0
        i = 0
        while i < len(self.tokenized_code):
            token = self.tokenized_code[i]
            if token == "#":
                while comment_lengths[comment_idx] > 0:
                    comment_lengths[comment_idx] -= 1
                    i += 1
                comment_idx += 1
            token = self.tokenized_code[i]
            new_tokens.append(token)
            i += 1
        self.tokenized_code = new_tokens

    def mask_variable_names(self, prefix="var") -> str:
        for i, variable in enumerate(self.variables):
            for loc in variable.locations:
                self.tokenized_code[loc] = f"{prefix}_{i}"


def get_variable_locations(
    variables: list[str], source_tokens: list[str]
) -> list[list[int]]:
    results = {variable: [] for variable in variables}

    for i, token in enumerate(source_tokens):
        if token in variables:
            results[token].append(i)

    return [results[variable] for variable in variables]


def get_func_def_locations(
    functions: list[str], source_tokens: list[str]
) -> list[list[int]]:
    results = {func: [] for func in functions}

    for i, token in enumerate(source_tokens):
        if token in functions or token.removeprefix("def").strip() in functions:
            results[token].append(i)

    return [results[func] for func in functions]
    return [results[func] for func in functions]
