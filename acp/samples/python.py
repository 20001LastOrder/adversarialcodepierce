import ast
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

    def visit_FunctionDef(self, node):
        # Record function names to ignore them later
        logger.debug(f"Function name {node.name}")
        self.defined_functions.add(node.name)
        # self.variables.add(node.name)
        for arg in node.args.args:
            self.variables.add(arg.arg)
        # Visit function body for local variables
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

    def tokenize(self, code: str) -> list[str]:
        tokens = tokenize.tokenize(BytesIO(code.encode("utf-8")).readline)
        return [token.string for token in tokens if token.type != tokenize.ENCODING]


def get_variable_locations(
    variables: list[str], source_tokens: list[str]
) -> list[list[int]]:
    results = {variable: [] for variable in variables}

    for i, token in enumerate(source_tokens):
        if token in variables:
            results[token].append(i)

    return [results[variable] for variable in variables]
