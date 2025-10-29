# Keith Briggs 2025-07-10

from math import log1p, sqrt, pi as π


def poisson_point_process_generator(uniform01, λ):
    """Radial Poisson generator
    λ = average points per unit area
    returns point (r,θ) in 2d polar coordinates
    The RNG uniform01 must return uniform (0,] variates.
    """
    a, s, twoπ = -1.0 / π / λ, 0.0, 2.0 * π
    while True:
        s += log1p(-uniform01())
        yield sqrt(a * s), twoπ * uniform01()
