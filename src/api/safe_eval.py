import ast
import logging
import math
import operator

logger = logging.getLogger(__name__)

bin_ops = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Unary operations mapping
un_ops = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def checkmath(x, *args):
    # Check if the function exists in math module and isn't private
    if x not in [func for func in dir(math) if not func.startswith("__")]:
        raise SyntaxError(f"Unknown function {x}()")
    fun = getattr(math, x)
    return fun(*args)


def _eval(node):
    if isinstance(node, ast.Expression):
        logger.debug("Expr")
        return _eval(node.body)
    elif isinstance(node, ast.Constant):
        logger.info("Const")
        # Restrict to numeric constants only
        if not isinstance(node.value, (int, float)):
            raise SyntaxError("Only numeric constants are allowed")
        return node.value
    elif isinstance(node, ast.BinOp):
        logger.debug("BinOp")
        left = _eval(node.left)
        right = _eval(node.right)
        op = bin_ops.get(type(node.op))
        if op is None:
            raise SyntaxError(f"Unsupported binary operation: {type(node.op).__name__}")
        return op(left, right)
    elif isinstance(node, ast.UnaryOp):
        logger.debug("UpOp")
        operand = _eval(node.operand)
        op = un_ops.get(type(node.op))
        if op is None:
            raise SyntaxError(f"Unsupported unary operation: {type(node.op).__name__}")
        return op(operand)
    elif isinstance(node, ast.Call):
        # Ensure function is a simple name (e.g., sin, not math.sin)
        if not isinstance(node.func, ast.Name):
            raise SyntaxError("Only simple function calls are allowed")
        args = [_eval(arg) for arg in node.args]
        return checkmath(node.func.id, *args)
    else:
        raise SyntaxError(f"Unsupported node type: {type(node).__name__}")


def safe_eval(s):
    # Parse the input string into an AST
    tree = ast.parse(s, mode="eval")

    return _eval(tree)
