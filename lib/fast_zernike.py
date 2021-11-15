

from typing import Union

from numpy import ndarray, ones, sqrt, sin, cos, arctan2 as atan2
from sympy import Symbol


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------


def zernike_derivative_cartesian(
    m: int,
    n: int,
    x: Union[float, ndarray],
    y: Union[float, ndarray],
    wrt: Union[str, Symbol],
) -> Union[float, ndarray]:
   

    # Derivatives for j = 0
    if m == 0 and n == 0 and wrt == "x":
        if isinstance(x, ndarray):
            return 0 * ones(x.shape)
        return 0
    if m == 0 and n == 0 and wrt == "y":
        if isinstance(x, ndarray):
            return 0 * ones(x.shape)
        return 0

    # Derivatives for j = 1
    if m == -1 and n == 1 and wrt == "x":
        if isinstance(x, ndarray):
            return 0 * ones(x.shape)
        return 0
    if m == -1 and n == 1 and wrt == "y":
        if isinstance(x, ndarray):
            return 2 * ones(x.shape)
        return 2

    # Derivatives for j = 2
    if m == 1 and n == 1 and wrt == "x":
        if isinstance(x, ndarray):
            return 2 * ones(x.shape)
        return 2
    if m == 1 and n == 1 and wrt == "y":
        if isinstance(x, ndarray):
            return 0 * ones(x.shape)
        return 0

    # Derivatives for j = 3
    if m == -2 and n == 2 and wrt == "x":
        return 2 * sqrt(6) * y
    if m == -2 and n == 2 and wrt == "y":
        return 2 * sqrt(6) * x

    # Derivatives for j = 4
    if m == 0 and n == 2 and wrt == "x":
        return 4 * sqrt(3) * x
    if m == 0 and n == 2 and wrt == "y":
        return 4 * sqrt(3) * y

    # Derivatives for j = 5
    if m == 2 and n == 2 and wrt == "x":
        return 2 * sqrt(6) * x
    if m == 2 and n == 2 and wrt == "y":
        return -2 * sqrt(6) * y

    # Derivatives for j = 6
    if m == -3 and n == 3 and wrt == "x":
        return (
            6
            * sqrt(2)
            * sqrt(x ** 2 + y ** 2)
            * (x * sin(3 * atan2(y, x)) - y * cos(3 * atan2(y, x)))
        )
    if m == -3 and n == 3 and wrt == "y":
        return (
            6
            * sqrt(2)
            * sqrt(x ** 2 + y ** 2)
            * (x * cos(3 * atan2(y, x)) + y * sin(3 * atan2(y, x)))
        )

    # Derivatives for j = 7
    if m == -1 and n == 3 and wrt == "x":
        return 12 * sqrt(2) * x * y
    if m == -1 and n == 3 and wrt == "y":
        return sqrt(2) * (6 * x ** 2 + 18 * y ** 2 - 4)

    # Derivatives for j = 8
    if m == 1 and n == 3 and wrt == "x":
        return sqrt(2) * (18 * x ** 2 + 6 * y ** 2 - 4)
    if m == 1 and n == 3 and wrt == "y":
        return 12 * sqrt(2) * x * y

    # Derivatives for j = 9
    if m == 3 and n == 3 and wrt == "x":
        return (
            6
            * sqrt(2)
            * sqrt(x ** 2 + y ** 2)
            * (x * cos(3 * atan2(y, x)) + y * sin(3 * atan2(y, x)))
        )
    if m == 3 and n == 3 and wrt == "y":
        return (
            6
            * sqrt(2)
            * sqrt(x ** 2 + y ** 2)
            * (-x * sin(3 * atan2(y, x)) + y * cos(3 * atan2(y, x)))
        )

    # Derivatives for j = 10
    if m == -4 and n == 4 and wrt == "x":
        return 4 * sqrt(10) * y * (3 * x ** 2 - y ** 2)
    if m == -4 and n == 4 and wrt == "y":
        return 4 * sqrt(10) * x * (x ** 2 - 3 * y ** 2)

    # Derivatives for j = 11
    if m == -2 and n == 4 and wrt == "x":
        return 2 * sqrt(10) * y * (12 * x ** 2 + 4 * y ** 2 - 3)
    if m == -2 and n == 4 and wrt == "y":
        return 2 * sqrt(10) * x * (4 * x ** 2 + 12 * y ** 2 - 3)

    # Derivatives for j = 12
    if m == 0 and n == 4 and wrt == "x":
        return sqrt(5) * x * (24 * x ** 2 + 24 * y ** 2 - 12)
    if m == 0 and n == 4 and wrt == "y":
        return sqrt(5) * y * (24 * x ** 2 + 24 * y ** 2 - 12)

    # Derivatives for j = 13
    if m == 2 and n == 4 and wrt == "x":
        return 2 * sqrt(10) * x * (8 * x ** 2 - 3)
    if m == 2 and n == 4 and wrt == "y":
        return 2 * sqrt(10) * y * (3 - 8 * y ** 2)

    # Derivatives for j = 14
    if m == 4 and n == 4 and wrt == "x":
        return 4 * sqrt(10) * x * (x ** 2 - 3 * y ** 2)
    if m == 4 and n == 4 and wrt == "y":
        return 4 * sqrt(10) * y * (-3 * x ** 2 + y ** 2)

    # Derivatives for j = 15
    if m == -5 and n == 5 and wrt == "x":
        if isinstance(x, ndarray):
            return 0 * ones(x.shape)
        return 0
    if m == -5 and n == 5 and wrt == "y":
        if isinstance(x, ndarray):
            return 0 * ones(x.shape)
        return 0

    # Derivatives for j = 16
    if m == -3 and n == 5 and wrt == "x":
        return sqrt(3 * x ** 2 + 3 * y ** 2) * (
            2 * x * (25 * x ** 2 + 25 * y ** 2 - 12) * sin(3 * atan2(y, x))
            - 6 * y * (5 * x ** 2 + 5 * y ** 2 - 4) * cos(3 * atan2(y, x))
        )
    if m == -3 and n == 5 and wrt == "y":
        return sqrt(3 * x ** 2 + 3 * y ** 2) * (
            6 * x * (5 * x ** 2 + 5 * y ** 2 - 4) * cos(3 * atan2(y, x))
            + 2 * y * (25 * x ** 2 + 25 * y ** 2 - 12) * sin(3 * atan2(y, x))
        )

    # Derivatives for j = 17
    if m == -1 and n == 5 and wrt == "x":
        return sqrt(3) * x * y * (80 * x ** 2 + 80 * y ** 2 - 48)
    if m == -1 and n == 5 and wrt == "y":
        return sqrt(3) * (
            20 * x ** 4
            + 120 * x ** 2 * y ** 2
            - 24 * x ** 2
            + 100 * y ** 4
            - 72 * y ** 2
            + 6
        )

    # Derivatives for j = 18
    if m == 1 and n == 5 and wrt == "x":
        return sqrt(3) * (
            100 * x ** 4
            + 120 * x ** 2 * y ** 2
            - 72 * x ** 2
            + 20 * y ** 4
            - 24 * y ** 2
            + 6
        )
    if m == 1 and n == 5 and wrt == "y":
        return sqrt(3) * x * y * (80 * x ** 2 + 80 * y ** 2 - 48)

    # Derivatives for j = 19
    if m == 3 and n == 5 and wrt == "x":
        return sqrt(3 * x ** 2 + 3 * y ** 2) * (
            2 * x * (25 * x ** 2 + 25 * y ** 2 - 12) * cos(3 * atan2(y, x))
            + 6 * y * (5 * x ** 2 + 5 * y ** 2 - 4) * sin(3 * atan2(y, x))
        )
    if m == 3 and n == 5 and wrt == "y":
        return sqrt(3 * x ** 2 + 3 * y ** 2) * (
            -6 * x * (5 * x ** 2 + 5 * y ** 2 - 4) * sin(3 * atan2(y, x))
            + 2 * y * (25 * x ** 2 + 25 * y ** 2 - 12) * cos(3 * atan2(y, x))
        )

    # Derivatives for j = 20
    if m == 5 and n == 5 and wrt == "x":
        return (
            10
            * sqrt(3)
            * (x ** 2 + y ** 2) ** (3 / 2)
            * (x * cos(5 * atan2(y, x)) + y * sin(5 * atan2(y, x)))
        )
    if m == 5 and n == 5 and wrt == "y":
        return (
            10
            * sqrt(3)
            * (x ** 2 + y ** 2) ** (3 / 2)
            * (-x * sin(5 * atan2(y, x)) + y * cos(5 * atan2(y, x)))
        )

    # Derivatives for j = 21
    if m == -6 and n == 6 and wrt == "x":
        if isinstance(x, ndarray):
            return 0 * ones(x.shape)
        return 0
    if m == -6 and n == 6 and wrt == "y":
        if isinstance(x, ndarray):
            return 0 * ones(x.shape)
        return 0

    # Derivatives for j = 22
    if m == -4 and n == 6 and wrt == "x":
        return (
            4
            * sqrt(14)
            * y
            * (30 * x ** 4 - 15 * x ** 2 - 6 * y ** 4 + 5 * y ** 2)
        )
    if m == -4 and n == 6 and wrt == "y":
        return (
            4
            * sqrt(14)
            * x
            * (6 * x ** 4 - 5 * x ** 2 - 30 * y ** 4 + 15 * y ** 2)
        )

    # Derivatives for j = 23
    if m == -2 and n == 6 and wrt == "x":
        return (
            sqrt(14)
            * y
            * (
                150 * x ** 4
                + 180 * x ** 2 * y ** 2
                - 120 * x ** 2
                + 30 * y ** 4
                - 40 * y ** 2
                + 12
            )
        )
    if m == -2 and n == 6 and wrt == "y":
        return (
            sqrt(14)
            * x
            * (
                30 * x ** 4
                + 180 * x ** 2 * y ** 2
                - 40 * x ** 2
                + 150 * y ** 4
                - 120 * y ** 2
                + 12
            )
        )

    # Derivatives for j = 24
    if m == 0 and n == 6 and wrt == "x":
        return (
            sqrt(7)
            * x
            * (
                -120 * x ** 2
                - 120 * y ** 2
                + 120 * (x ** 2 + y ** 2) ** 2
                + 24
            )
        )
    if m == 0 and n == 6 and wrt == "y":
        return (
            sqrt(7)
            * y
            * (
                -120 * x ** 2
                - 120 * y ** 2
                + 120 * (x ** 2 + y ** 2) ** 2
                + 24
            )
        )

    # Derivatives for j = 25
    if m == 2 and n == 6 and wrt == "x":
        return (
            sqrt(14)
            * x
            * (
                90 * x ** 4
                + 60 * x ** 2 * y ** 2
                - 80 * x ** 2
                - 30 * y ** 4
                + 12
            )
        )
    if m == 2 and n == 6 and wrt == "y":
        return (
            sqrt(14)
            * y
            * (
                30 * x ** 4
                - 60 * x ** 2 * y ** 2
                - 90 * y ** 4
                + 80 * y ** 2
                - 12
            )
        )

    # Derivatives for j = 26
    if m == 4 and n == 6 and wrt == "x":
        return (
            4
            * sqrt(14)
            * x
            * (
                9 * x ** 4
                - 30 * x ** 2 * y ** 2
                - 5 * x ** 2
                - 15 * y ** 4
                + 15 * y ** 2
            )
        )
    if m == 4 and n == 6 and wrt == "y":
        return (
            4
            * sqrt(14)
            * y
            * (
                -15 * x ** 4
                - 30 * x ** 2 * y ** 2
                + 15 * x ** 2
                + 9 * y ** 4
                - 5 * y ** 2
            )
        )

    # Derivatives for j = 27
    if m == 6 and n == 6 and wrt == "x":
        return (
            6
            * sqrt(14)
            * (x ** 2 + y ** 2) ** 2
            * (x * cos(6 * atan2(y, x)) + y * sin(6 * atan2(y, x)))
        )
    if m == 6 and n == 6 and wrt == "y":
        return (
            6
            * sqrt(14)
            * (x ** 2 + y ** 2) ** 2
            * (-x * sin(6 * atan2(y, x)) + y * cos(6 * atan2(y, x)))
        )

    # Derivatives for j = 28
    if m == -7 and n == 7 and wrt == "x":
        if isinstance(x, ndarray):
            return 0 * ones(x.shape)
        return 0
    if m == -7 and n == 7 and wrt == "y":
        if isinstance(x, ndarray):
            return 0 * ones(x.shape)
        return 0

    # Derivatives for j = 29
    if m == -5 and n == 7 and wrt == "x":
        if isinstance(x, ndarray):
            return 0 * ones(x.shape)
        return 0
    if m == -5 and n == 7 and wrt == "y":
        if isinstance(x, ndarray):
            return 0 * ones(x.shape)
        return 0

    # Derivatives for j = 30
    if m == -3 and n == 7 and wrt == "x":
        return sqrt(x ** 2 + y ** 2) * (
            x
            * (
                -600 * x ** 2
                - 600 * y ** 2
                + 588 * (x ** 2 + y ** 2) ** 2
                + 120
            )
            * sin(3 * atan2(y, x))
            - 3
            * y
            * (-120 * x ** 2 - 120 * y ** 2 + 84 * (x ** 2 + y ** 2) ** 2 + 40)
            * cos(3 * atan2(y, x))
        )
    if m == -3 and n == 7 and wrt == "y":
        return sqrt(x ** 2 + y ** 2) * (
            3
            * x
            * (-120 * x ** 2 - 120 * y ** 2 + 84 * (x ** 2 + y ** 2) ** 2 + 40)
            * cos(3 * atan2(y, x))
            + y
            * (
                -600 * x ** 2
                - 600 * y ** 2
                + 588 * (x ** 2 + y ** 2) ** 2
                + 120
            )
            * sin(3 * atan2(y, x))
        )

   
    # Raise value error if we have not returned yet
    raise ValueError(
        "No pre-computed derivative available for the given arguments!"
    )
