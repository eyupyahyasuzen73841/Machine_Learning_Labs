"""
Core Scalar Automatic Differentiation Framework
Student Researcher: Eyüp Yahya Süzen (ID: 73841)
Curriculum: Machine Learning Systems (Lab 05)
Institution: Vistula University
"""

from __future__ import annotations
import math
from typing import Tuple, Set, Callable, Optional, Union, List


class GradScalar:
    """
    Core graph node encapsulating a scalar value, incoming gradient,
    parent references, and local backward differentiation step.
    """
    __slots__ = ('value', 'grad', 'parents', 'backward_step', 'op', 'name')

    def __init__(
        self,
        value: Union[float, int],
        parents: Tuple[GradScalar, ...] = (),
        op: str = '',
        name: Optional[str] = None
    ) -> None:
        self.value: float = float(value)
        self.grad: float = 0.0
        self.parents: Set[GradScalar] = set(parents)
        self.backward_step: Callable[[], None] = lambda: None
        self.op: str = op
        self.name: Optional[str] = name

    def __repr__(self) -> str:
        ident = f" [{self.name}]" if self.name else ""
        return f"GradScalar(value={self.value:.4f}, grad={self.grad:.4f}, op='{self.op}'{ident})"

    def __add__(self, other: Union[GradScalar, float, int]) -> GradScalar:
        rhs = other if isinstance(other, GradScalar) else GradScalar(other)
        out = GradScalar(self.value + rhs.value, (self, rhs), '+')

        def _step() -> None:
            self.grad += 1.0 * out.grad
            rhs.grad += 1.0 * out.grad

        out.backward_step = _step
        return out

    def __radd__(self, other: Union[float, int]) -> GradScalar:
        return self.__add__(other)

    def __mul__(self, other: Union[GradScalar, float, int]) -> GradScalar:
        rhs = other if isinstance(other, GradScalar) else GradScalar(other)
        out = GradScalar(self.value * rhs.value, (self, rhs), '*')

        def _step() -> None:
            self.grad += rhs.value * out.grad
            rhs.grad += self.value * out.grad

        out.backward_step = _step
        return out

    def __rmul__(self, other: Union[float, int]) -> GradScalar:
        return self.__mul__(other)

    def __pow__(self, exponent: Union[float, int]) -> GradScalar:
        if not isinstance(exponent, (int, float)):
            raise TypeError("Exponent must be integer or floating-point number")
        out = GradScalar(self.value ** exponent, (self,), f'**{exponent}')

        def _step() -> None:
            self.grad += (exponent * (self.value ** (exponent - 1))) * out.grad

        out.backward_step = _step
        return out

    def relu(self) -> GradScalar:
        val = self.value if self.value > 0.0 else 0.0
        out = GradScalar(val, (self,), 'ReLU')

        def _step() -> None:
            self.grad += (1.0 if self.value > 0.0 else 0.0) * out.grad

        out.backward_step = _step
        return out

    def sigmoid(self) -> GradScalar:
        # Logistic sigmoid with numerical stability protection
        if self.value >= 0.0:
            e = math.exp(-self.value)
            res = 1.0 / (1.0 + e)
        else:
            e = math.exp(self.value)
            res = e / (1.0 + e)
        out = GradScalar(res, (self,), 'Sigmoid')

        def _step() -> None:
            self.grad += (res * (1.0 - res)) * out.grad

        out.backward_step = _step
        return out

    def log(self, eps: float = 1e-12) -> GradScalar:
        clamped = max(self.value, eps)
        out = GradScalar(math.log(clamped), (self,), 'Log')

        def _step() -> None:
            self.grad += (1.0 / clamped) * out.grad

        out.backward_step = _step
        return out

    def __neg__(self) -> GradScalar:
        return self * -1.0

    def __sub__(self, other: Union[GradScalar, float, int]) -> GradScalar:
        return self + (-other)

    def __rsub__(self, other: Union[float, int]) -> GradScalar:
        return GradScalar(other) + (-self)

    def __truediv__(self, other: Union[GradScalar, float, int]) -> GradScalar:
        rhs = other if isinstance(other, GradScalar) else GradScalar(other)
        return self * (rhs ** -1.0)

    def backward(self) -> None:
        """
        Executes reverse-mode automatic differentiation:
        1. Topologically sorts the computation graph.
        2. Sets initial output gradient to 1.0.
        3. Traverses nodes in reverse order, accumulating gradients via +=.
        """
        topo_order: List[GradScalar] = []
        visited: Set[GradScalar] = set()

        def dfs(node: GradScalar) -> None:
            if node not in visited:
                visited.add(node)
                for parent in node.parents:
                    dfs(parent)
                topo_order.append(node)

        dfs(self)
        self.grad = 1.0

        for current in reversed(topo_order):
            current.backward_step()
