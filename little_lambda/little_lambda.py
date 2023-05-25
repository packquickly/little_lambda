from typing import Any, Callable, TYPE_CHECKING
from jaxtyping import Array
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu


class _Metaλ(type):
    def __mul__(cls, value):
        return cls(value)


class _λ(metaclass=_Metaλ):
    """Provides a functional-like syntax for function composition and partial function
    application.

    !!! example
    λ(jnp.sin) * jnp.sum = lambda x: jnp.sin(jnp.sum(x))

    λ also support the equivalent "chain" notation:
    !!! example
    ```python
    λ * f * g = lambda x: f(g(x))
    ```

    λ can also perform partial application and addition:
    !!! example
    ```python
    λ(f, y) + g = lambda x: f(y, x) + g(x)
    ```

    This will promote numerical types to functions, ie.
    ```python
    λ * 17 * f = lambda x: 17 * f(x)
    ```
    """

    def __init__(self, fn, /, *args, **kwargs):
        """**Arguments:**

        - `fn`: the callable to partially apply
        - `*args`: any positional arguments to apply.
        - `**kwargs`: any keyword arguments to apply.
        """
        self.λ = eqx.Partial(fn, *args, **kwargs)

    def __repr__(self):
        return f"λ({self.λ})"

    def __mul__(self, other):
        if isinstance(other, Callable):
            fn = other
        elif isinstance(other, (complex, float, int, jax.Array)):
            fn = lambda x: other * x
        else:
            RuntimeError("Type of `other` not understood.")

        def composition(*args, **kwargs):
            return self.λ(fn(*args, **kwargs))

        return λ(composition)

    def __rmul__(self, other):
        if isinstance(other, Callable):
            fn = other
        elif isinstance(other, (complex, float, int, Array)):
            fn = lambda x: other * x
        else:
            RuntimeError("Type of `other` not understood.")

        def composition(*args, **kwargs):
            return fn(self.λ(*args, **kwargs))

        return λ(composition)

    def __add__(self, other):
        if isinstance(other, λ):

            def fun_sum(*args, **kwargs):  # pyright: ignore
                return self.λ(*args, **kwargs) + other.λ(*args, **kwargs)

        elif isinstance(other, (complex, float, int, jax.Array)):

            def fun_sum(*args, **kwargs):
                return self.λ(*args, **kwargs) + other

        else:
            raise RuntimeError("Type of `other` not understood.")
        return λ(fun_sum)

    def __call__(self, *args, **kwargs):
        return self.λ(*args, **kwargs)


if TYPE_CHECKING:
    λ = Any
else:
    λ = _λ
