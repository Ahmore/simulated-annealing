import argparse
from random import random, randrange
from math import fmod, sqrt

import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    raise

import utils



#####################
# Points generators #
#####################

def generate(n, min_x, max_x, min_y, max_y):
    x = []
    y = []

    for i in range(0, n):
        x.append(random()*(max_x - min_x) + min_x)
        y.append(random()*(max_y - min_y) + min_y)

    return np.array([x, y]).transpose()


def generate_gauss(n, mu, sigma):
    x = np.random.normal(mu, sigma, n)
    y = np.random.normal(mu, sigma, n)

    return np.array([x, y]).transpose()


def generate_cloud(n, min_x, max_x, min_y, max_y):
    x = [
        [min_x, (max_x - min_x)/9],
        [4*(max_x - min_x)/9, 5*(max_x - min_x)/9],
        [8*(max_x - min_x)/9, max_x]
    ]

    y = [
        [min_y, (max_y - min_y)/9],
        [4*(max_y - min_y)/9, 5*(max_y - min_y)/9],
        [8*(max_y - min_y)/9, max_y]
    ]

    a = []
    for i in range(0, 3):
        for j in range(0, 3):
            arr = generate(int(n/9), x[i][0], x[i][1], y[j][0], y[j][1])

            for k in range(0, arr.__len__()):
                a.append(arr[k])

    return np.array(a)



######################
# Neighbor functions #
######################

def n_arbitrary(a):
    rands = []
    (m, n) = a.shape

    # Generate elements to swap
    for i in range(0, 2):
        rands.append(randrange(0, m))

    # Swap
    a[[rands[0], rands[1]]] = a[[rands[1], rands[0]]]

    return a


def n_consecutive(a):
    (m, n) = a.shape

    # Random element to consider
    r = randrange(0, m)

    # Random direction
    d = randrange(0, 1)

    # Set nodes to swap
    if (d == 0):
        s = fmod(r - 1, m)
    else:
        s = fmod(r + 1, m)

    # Swap
    a[[r, s]] = a[[s, r]]

    return a



###################
# Energy function #
###################

def energy(a):
    (m, n) = a.shape
    e = 0

    for i in range(0, m-1):
        e += sqrt((a[i+1][0]-a[i][0])**2+(a[i+1][1]-a[i][1])**2)

    return e



f_gen_dict = {
    "steady": generate,
    "gauss": generate_gauss,
    "cloud": generate_cloud
}

t_fun_dict = {
    "log": utils.t_log,
    "lin": utils.t_lin,
    "pow": utils.t_pow
}

n_fun_dict = {
    "a": n_arbitrary,
    "c": n_consecutive
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--f_gen", type=str, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--t0", type=float, required=True)
    parser.add_argument("--t_fun", type=str, required=True)
    parser.add_argument("--n_fun", type=str, required=True)
    parser.add_argument("--max", type=int, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--mu", type=int)
    parser.add_argument("--sigma", type=int)
    parser.add_argument("--eps", type=float, required=True)
    args = parser.parse_args()

    plt.gca().invert_yaxis()

    if args.f_gen == "gauss":
        arr = generate_gauss(args.n, args.mu, args.sigma)
    else:
        arr = f_gen_dict[args.f_gen](args.n, 0, 1000, 0, 1000)

    e = utils.s_a(arr, args.t0, t_fun_dict[args.t_fun], n_fun_dict[args.n_fun], energy, utils.plot_xy, utils.plot_xy, args.max, args.eps, args.name)
    print(e)
