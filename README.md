# little λ
little λ is a tiny python utility for function composition!

little λ supports a functional-like syntax for function composition by
extending the `@` operator used for linear function composition to
nonlinear functions via the `λ` class.

To use, simply put a little `λ` at the start of the functions
you want to compose separated by composition operators `@`:

```python
n_unique = λ @ len @ set
n_unique([1, 1, 4, 5])
# returns 3
```

The `λ` can also be used to generate simple lambda functions from
most binary operations, which can themselves be composed:

```python
is_even = (λ == 0) @ (λ % 2)
```

Note that the term `(λ == 0)` counts as a `λ` at the start of the
composition.

Finally, little `λ` can be used for partial application calling `λ` with the
function in the first argument, and the arguments to apply in the following arguments:

```python
sum_of_squares = λ @ sum @ λ(map, λ ** 2)
```

Here, `λ(map, λ ** 2)` is the map function, partially applied with the square function.


little λ makes your functional Python code look the part!
```python
sum_of_even_squares = λ @ sum @ λ(map, λ ** 2) @ λ(filter, (λ == 0) @ (λ % 2))
```


# WARNING
little λ is not a maintained package and is not intended for serious use.
It is by design not Pythonic, and it likely has many edge-cases I'm unaware of,
and may not ever fix.
