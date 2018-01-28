from math import log, exp
from random import random
import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    raise


def s_a(points, t0, t_fun, n_fun, e_fun, plot_solution, plot_energy, max, eps, filename):
    # Plot before changes
    plot_solution(points, filename + "_before")

    # Init min path
    p_min = np.array(points, copy=True)
    p_min_energy = e_fun(p_min)

    # Init start data
    e = []
    t = t0
    a = np.array(points, copy=True)
    a_energy = e_fun(a)
    i = 1

    while i < max and t > eps:
        n = n_fun(np.array(a, copy=True))
        n_energy = e_fun(n)

        # If its better state
        if n_energy < a_energy:
            a = n
            a_energy = n_energy

            # If is better than the best
            if a_energy < p_min_energy:
                p_min = a
                p_min_energy = a_energy

        # With some probability get worse state
        elif exp(-1*(n_energy - a_energy)/t) > random():
            a = n
            a_energy = n_energy

        # Remember energy changes
        e.append([i, a_energy])

        # Count new temperature
        t = t_fun(t0, i, max)

        i += 1

    # Plot energies changes
    plot_energy(e, filename + "_energy")

    # Plot solution
    plot_solution(p_min, filename + "_after")

    return p_min_energy


def s_a2(points, start, t0, t_fun, n_fun, e_fun, plot_solution, plot_energy, max, eps, filename, e_min):
    # Plot before changes
    # plot_solution(points, filename + "_before")

    # Init min path
    p_min = np.array(points, copy=True)
    p_min_energy = e_fun(p_min)

    # Init start data
    e = []
    t = t0
    a = np.array(points, copy=True)
    a_energy = e_fun(a)
    i = 1

    while i < max and t > eps and a_energy > e_min:
        n = n_fun(np.array(a, copy=True), start)
        n_energy = e_fun(n)

        # If its better state
        if n_energy < a_energy:
            a = n
            a_energy = n_energy

            # If is better than the best
            if a_energy < p_min_energy:
                p_min = a
                p_min_energy = a_energy

        # With some probability get worse state
        elif exp(-1*(n_energy - a_energy)/t) > random():
            a = n
            a_energy = n_energy

        # Remember energy changes
        e.append([i, a_energy])

        # Count new temperature
        t = t_fun(t0, i, max)

        i += 1

    # Plot energies changes
    # plot_energy(e, filename + "_energy")

    # Plot solution
    # plot_solution(p_min, filename + "_after")

    return i


def t_pow(t, i, max):
    return t*(0.999**i)


def t_log(t, i, max):
    return t*(log(max)-log(i))


def t_lin(t, i, max):
    return t - i*(t/max)


def plot_xy(a, filename):
    plt.figure()
    a = np.array(a)
    plt.plot(a[:, 0], a[:, 1], linewidth=.5, ms=.5)
    plt.savefig(filename + ".png")

