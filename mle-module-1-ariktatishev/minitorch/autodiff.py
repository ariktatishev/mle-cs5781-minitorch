from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # Task 1.1.
    new_vals = list(vals)
    pos_eps = new_vals.copy()
    neg_eps = new_vals.copy()
    pos_eps[arg] += epsilon
    neg_eps[arg] -= epsilon
    return (f(*pos_eps) - f(*neg_eps)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """

    # Task 1.4.
    def dfs(variable: Variable, output: List[Variable], visited: Any) -> None:
        if variable.unique_id in visited:
            return

        if not variable.is_leaf():
            for parent in variable.parents:
                if not parent.is_constant():
                    dfs(parent, output, visited)

        output.insert(0, variable)
        visited.append(variable.unique_id)

    output: List[Variable] = list([])
    visited: List[Variable] = list([])
    dfs(variable, output, visited)
    return output


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # Task 1.4.

    assert not variable.is_constant(), "Can't call backward on a constant."
    # topological sort
    variables = topological_sort(variable)
    scalarAndDerivatives = {}
    for var in variables:
        scalarAndDerivatives[var.unique_id] = 0.0

    scalarAndDerivatives[variable.unique_id] = deriv

    for var in variables:
        if var.is_leaf():
            var.accumulate_derivative(scalarAndDerivatives[var.unique_id])
        else:
            for v, derivative in var.chain_rule(scalarAndDerivatives[var.unique_id]):
                scalarAndDerivatives[v.unique_id] += derivative


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
