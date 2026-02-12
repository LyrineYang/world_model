from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Mapping


class StrategyError(ValueError):
    pass


_ALLOWED_NODE_TYPES = (
    ast.Expression,
    ast.BoolOp,
    ast.UnaryOp,
    ast.BinOp,
    ast.Compare,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.Eq,
    ast.NotEq,
    ast.Gt,
    ast.GtE,
    ast.Lt,
    ast.LtE,
)


def _validate_tree(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            raise StrategyError(f"Unsupported syntax in strategy: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise StrategyError(f"Illegal variable name in strategy: {node.id}")


@dataclass(frozen=True)
class CompiledStrategy:
    expression: str
    _code: Any

    def evaluate(self, context: Mapping[str, Any]) -> bool:
        try:
            result = eval(self._code, {"__builtins__": {}}, dict(context))
        except NameError as exc:
            raise StrategyError(f"Unknown variable in strategy: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise StrategyError(f"Strategy evaluation failed: {exc}") from exc

        if isinstance(result, bool):
            return result
        if isinstance(result, (int, float)):
            return bool(result)
        raise StrategyError(f"Strategy must evaluate to bool/number, got {type(result).__name__}")


def compile_strategy(expression: str) -> CompiledStrategy:
    expr = (expression or "").strip()
    if not expr:
        raise StrategyError("Empty strategy expression")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise StrategyError(f"Invalid strategy syntax: {exc}") from exc
    _validate_tree(tree)
    code = compile(tree, "<strategy>", "eval")
    return CompiledStrategy(expression=expr, _code=code)

