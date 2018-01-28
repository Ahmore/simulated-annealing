import argparse
from random import random, randrange
from math import fmod
import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    raise
import utils



#####################
# Points generators #
#####################

def generate(n, d):
    a = np.zeros((n, n))

    for i in range(0, n):
        for j in range(0, n):
            if d > random():
                a[i][j] = 1

    return a



######################
# Neighbor functions #
######################


def n_arbitrary(a):
    rands = []
    (m, n) = a.shape

    for i in range(0, 4):
        rands.append(randrange(0, m))

    a[rands[0]][rands[1]], a[rands[2]][rands[3]] = a[rands[2]][rands[3]], a[rands[0]][rands[1]]

    return a


def n_consecutive(a):
    rands = []
    (m, n) = a.shape

    for i in range(0, 2):
        rands.append(randrange(0, m))

    for i in range(0, 2):
        rands.append(int(fmod(rands[i] + randrange(-1, 1), m)))

    a[rands[0]][rands[1]], a[rands[2]][rands[3]] = a[rands[2]][rands[3]], a[rands[0]][rands[1]]

    return a



####################
# Energy functions #
####################

def e_fun_cross(w_fun):
    def f(a):
        dir = [
            [1, 1],
            [1, -1],
            [-1, -1],
            [-1, 1]
        ]

        return e_iter(a, dir, w_fun)
    return f


def e_fun_ring(w_fun):
    def f(a):
        dir = [
            [1, 1],
            [1, -1],
            [-1, -1],
            [-1, 1],
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, -1]
        ]

        return e_iter(a, dir, w_fun)
    return f

def e_fun_big_ring(w_fun):
    def f(a):
        dir = [
            [-2, 2],
            [-2, 1],
            [-2, 0],
            [-2, -1],
            [-2, -2],
            [-1, -2],
            [0, -2],
            [1, -2],
            [2, -2],
            [2, -1],
            [2, 0],
            [2, 1],
            [2, 2],
            [1, 2],
            [0, 2],
            [-1, 2]
        ]

        return e_iter(a, dir, w_fun)
    return f

def e_fun_slash(w_fun):
    def f(a):
        dir = [
            [-2, 2],
            [-1, 1],
            [1, -1],
            [2, -2]
        ]

        return e_iter(a, dir, w_fun)
    return f


def e_fun_square(w_fun):
    def f(a):
        dir = [
            [-2, 2],
            [-2, 1],
            [-2, 0],
            [-2, -1],
            [-2, -2],
            [-1, -2],
            [0, -2],
            [1, -2],
            [2, -2],
            [2, -1],
            [2, 0],
            [2, 1],
            [2, 2],
            [1, 2],
            [0, 2],
            [-1, 2],
            [1, 1],
            [1, -1],
            [-1, -1],
            [-1, 1],
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, -1]
        ]

        return e_iter(a, dir, w_fun)
    return f

def e_iter(a, dir, w_fun):
    (m, n) = a.shape
    l = dir.__len__()
    e = 0

    # Over every cell
    for i in range(0, m):
        for j in range(0, m):
            for k in range(0, l):
                x = int(fmod(j + dir[k][0], m))
                y = int(fmod(i + dir[k][1], m))

                e += w_fun(a, i, j, x, y)
    return e



######################
# Relation functions #
######################

def w_fun1(a, i, j, x, y):
    e = 0

    # White
    if a[i][j] == 0:
        # Doesnt like black
        if a[y][x] == 1:
            e = 1

    # Black
    else:
        # Doesnt like white
        if a[y][x] == 0:
            e = 1

    return e

# Doesnt like same color
def w_fun2(a, i, j, x, y):
    e = 0

    # White
    if a[i][j] == 0:
        # Doesnt like black
        if a[y][x] == 0:
            e = 1

    # Black
    else:
        # Doesnt like white
        if a[y][x] == 1:
            e = 1

    return e

# White doesnt like black on right
# Black doesnt like white on left
def w_fun3(a, i, j, x, y):
    e = 0

    # White
    if a[i][j] == 0:
        if x > j and a[y][x] == 1:
            e = 1

    # Black
    else:
        if x < j and a[y][x] == 0:
            e = 1

    return e

# White doesnt like black above
# Black doesnt like white belove
def w_fun4(a, i, j, x, y):
    e = 0

    # White
    if a[i][j] == 0:
        if y > i and a[y][x] == 1:
            e = 1

    # Black
    else:
        if y < i and a[y][x] == 0:
            e = 1

    return e

def plot_solution(a, filename):
    plt.figure()
    plt.imshow(a, cmap='Greys')
    plt.savefig(filename + ".png")


t_fun_dict = {
    "log": utils.t_log,
    "lin": utils.t_lin,
    "pow": utils.t_pow
}

e_fun_dict = {
    "cross1": e_fun_cross(w_fun1),
    "cross2": e_fun_cross(w_fun2),
    "ring1": e_fun_ring(w_fun1),
    "ring3": e_fun_ring(w_fun3),
    "ring4": e_fun_ring(w_fun4),
    "big_ring1": e_fun_big_ring(w_fun1),
    "big_ring2": e_fun_big_ring(w_fun2),
    "slash2": e_fun_slash(w_fun2)
}

n_fun_dict = {
    "a": n_arbitrary,
    "c": n_consecutive
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--d", type=float, required=True)
    parser.add_argument("--t0", type=float, required=True)
    parser.add_argument("--t_fun", type=str, required=True)
    parser.add_argument("--n_fun", type=str, required=True)
    parser.add_argument("--e_fun", type=str, required=True)
    parser.add_argument("--max", type=int, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--eps", type=float, required=True)
    args = parser.parse_args()

    plt.gca().invert_yaxis()

    arr = generate(args.n, args.d)

    e = utils.s_a(arr, args.t0, t_fun_dict[args.t_fun], n_fun_dict[args.n_fun], e_fun_dict[args.e_fun], plot_solution, utils.plot_xy, args.max, args.eps, args.name)
    print(e)
