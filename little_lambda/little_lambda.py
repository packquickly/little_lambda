import functools as ft
from typing import Any, Callable, TYPE_CHECKING
from jaxtyping import Array
import operator
import jax
import jax.numpy as jnp
import jax.tree_util as jtu


class _Metaλ(type):
    def __matmul__(cls, value):
        return cls(value)


class _λ(metaclass=_Metaλ):
    """Provides a functional-like syntax for function composition and partial function
    application.

    !!! example
    ```python
    λ(f) @ λ(g) = lambda x: f(g(x))
    ```

    λ also supports the equivalent "chain" notation:
    !!! example
    ```python
    λ @ f @ g = lambda x: f(g(x))
    ```

    λ will perform partial application when multiple arguments are supplied,
    and supports addition of functions:
    !!! example
    ```python
    λ(f, y) + g = lambda x: f(y, x) + g(x)
    ```

    and can partially apply binary operations to constants:
    !!! example
    ```python
    (λ + 2) = lamda x: x + 2
    (λ * jnp.ones(182)) = lmabda x: x * jnp.ones(182)
    ```
    """

    def __init__(self, fn, /, *args, **kwargs):
        """**Arguments:**

        - `fn`: the callable to partially apply
        - `*args`: any positional arguments to apply.
        - `**kwargs`: any keyword arguments to apply.
        """
        self.λ = ft.partial(fn, *args, **kwargs)

    def __repr__(self):
        return f"λ({self.λ})"

    def __matmul__(self, other):
        if isinstance(other, Callable):
            fn = other
        else:
            RuntimeError("Type of `other` not understood.")

        def composition(*args, **kwargs):
            return self.λ(fn(*args, **kwargs))

        return λ(composition)

    def __add__(self, other):
        if isinstance(other, λ):

            def summation(*args, **kwargs):  # pyright: ignore
                return self.λ(*args, **kwargs) + other.λ(*args, **kwargs)

        else:
            raise RuntimeError("Type of `other` not understood.")
        return λ(summation)

    def __call__(self, *args, **kwargs):
        return self.λ(*args, **kwargs)

    
def _set_meta_binary(cls, name: str, op: Callable[[Any, Any], Any]) -> None:
    def fn(cls, value):            
        if isinstance(value, (bool, complex, float, int, jax.Array)):
            return cls(lambda x: op(x, value))
        else:                      
            raise RuntimeError("Type of `value` not understood.") 

    fn.__name__ = name
    fn.__qualname__ = cls.__qualname__ + "." + name
    setattr(cls, name, fn)

def _rev(op):
    def __rev(x, y):
        return op(y, x)
    return __rev

for (name, op) in [
    ("__add__", operator.add),
    ("__sub__", operator.sub),
    ("__mul__", operator.mul),
    ("__matmul__", operator.matmul),
    ("__truediv__", operator.truediv),
    ("__floordiv__", operator.floordiv),
    ("__mod__", operator.mod),
    ("__pow__", operator.pow),
    ("__lshift__", operator.lshift),
    ("__rshift__", operator.rshift),
    ("__and__", operator.and_),
    ("__xor__", operator.xor),
    ("__or__", operator.or_),
    ("__radd__", _rev(operator.add)),
    ("__rsub__", _rev(operator.sub)),
    ("__rmul__", _rev(operator.mul)),
    ("__rmatmul__", _rev(operator.matmul)),
    ("__rtruediv__", _rev(operator.truediv)),
    ("__rfloordiv__", _rev(operator.floordiv)),
    ("__rmod__", _rev(operator.mod)),
    ("__rpow__", _rev(operator.pow)),
    ("__rlshift__", _rev(operator.lshift)),
    ("__rrshift__", _rev(operator.rshift)),
    ("__rand__", _rev(operator.and_)),
    ("__rxor__", _rev(operator.xor)),
    ("__ror__", _rev(operator.or_)),
    ("__lt__", operator.lt),
    ("__le__", operator.le),
    ("__eq__", operator.eq),
    ("__ne__", operator.ne),
    ("__gt__", operator.gt),
    ("__ge__", operator.ge),
]:
    _set_meta_binary(_Metaλ, name, op)

if TYPE_CHECKING:
    λ = Any
else:
    λ = _λ

def add(x, y):
    return x + y


if __name__ == "__main__":
    two_norm_lambda = lambda tree: jtu.tree_reduce(
        add, jtu.tree_map(lambda x: jnp.sum(x**2), tree)
    )
    two_norm_λ = λ(jtu.tree_reduce, add) @ λ(jtu.tree_map, λ @ jnp.sum @ jnp.square)
    ssa_lambda = lambda tree: jtu.tree_reduce(
        add,
        43
        * jtu.tree_map(
            lambda x: jnp.sum(jnp.square(jnp.sin(jnp.sqrt(jnp.abs(x))))), tree
        ),
    )
    ssa_λ = (
        λ(jtu.tree_reduce, add)
        @ 43
        @ λ(jtu.tree_map, λ @ jnp.sum @ jnp.square @ jnp.sin @ jnp.sqrt @ jnp.abs)
    )
    y = ({"a": (jnp.ones(3), 2 * jnp.ones(14))}, jnp.zeros(43))
    vector = jnp.zeros(10,)
    vec_λ = λ @ jnp.sum @ jnp.square @ (λ + 2)
    two_lambda = two_norm_lambda(y)
    two_λ = two_norm_λ(y)
    ssa_lambda = ssa_lambda(y)
    ssa_λ = ssa_λ(y)
    print(f"vec_λ is {vec_λ(vector)}")
    print(
        f"lambda 2 norm: {two_lambda}, lambda ssa: {ssa_lambda}",
        f"λ 2 norm ssa: {two_λ}, λ ssa: {ssa_λ}",
    )
