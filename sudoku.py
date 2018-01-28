import argparse
from random import random, randrange, sample
from math import fmod
import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    raise
import utils



##################
# Init functions #
##################

def load(filename):
    arr = np.loadtxt(args.input, dtype=str)
    arr = np.array(list(map(lambda x: list(map(lambda y: int(0 if y == 'x' else y), x)), arr)))

    return arr

def fill(a, original):
    for i in range(0, 3):
        for j in range(0, 3):
            indices1 = get_block_indices(i, j, original, False)
            numbers = np.zeros(9)

            for k in range(0,  indices1.__len__()):
                y, x = indices1[k]

                if a[y][x] != 0:
                    numbers[a[y][x]-1] = 1

            indices2 = get_block_indices(i, j, original, True)
            num_i = 0
            for k in range(0,  indices2.__len__()):
                y, x = indices2[k]

                while numbers[num_i] == 1:
                    num_i += 1

                a[y][x] = num_i + 1
                num_i += 1

    return a



####################
# Energy functions #
####################

def energy(a):
    return energy_block(a) + energy_row(a) + energy_col(a)


def energy_block(a):
    e = 0
    for i in range(0, 3):
        for j in range(0, 3):
            repetitions = np.zeros(9)

            for k in range(i*3, (i+1)*3):
                for l in range(j*3, (j+1)*3):
                    repetitions[a[k][l]-1] += 1

            for m in range(0, 9):
                if repetitions[m] > 1:
                    e += repetitions[m] - 1

    return e


def energy_row(a):
    e = 0

    for i in range(0, 9):
        repetitions = np.zeros(9)

        for j in range(0, 9):
            repetitions[a[i][j]-1] += 1

        for j in range(0, 9):
            if repetitions[j] > 1:
                e += repetitions[j] - 1

    return e


def energy_col(a):
    e = 0

    for i in range(0, 9):
        repetitions = np.zeros(9)

        for j in range(0, 9):
            repetitions[a[j][i]-1] += 1

        for j in range(0, 9):
            if repetitions[j] > 1:
                e += repetitions[j] - 1

    return e



######################
# Neighbor functions #
######################

def n_arbitrary(a, start):
    rands = []
    (m, n) = a.shape

    # Prevent change fixed numbers
    while rands.__len__() < 4:
        r = [randrange(0, m), randrange(0, m)]

        if start[r[0]][r[1]] == 0:
            rands.append(r[0])
            rands.append(r[1])

    a[rands[0]][rands[1]], a[rands[2]][rands[3]] = a[rands[2]][rands[3]], a[rands[0]][rands[1]]

    return a


def n_squares(a, origin):
    i = randrange(0, 3)
    j = randrange(0, 3)

    indices = get_block_indices(i, j, origin, False)
    num_in_block = len(indices)

    if num_in_block > 2:
        random_squares = sample(range(num_in_block), 2)
        square1, square2 = [indices[ind] for ind in random_squares]
        a[square1[0]][square1[1]], a[square2[0]][square2[1]] = a[square2[0]][square2[1]], a[square1[0]][square1[1]]

    return a



##################
# Plot functions #
##################

def plot_solution(a, filename):
    np.savetxt(filename + ".npy", a, fmt='%i')



###################
# Other functions #
###################

def get_block_indices(i, j, original, filter_by_originals=True):
    ys = [k for k in range(i*3, (i+1)*3)]
    xs = [l for l in range(j*3, (j+1)*3)]

    indices = [(y, x) for y in ys for x in xs]

    if filter_by_originals:
        indices = list(filter(lambda el: (original[el[0]][el[1]] == 0), indices))

    return indices



t_fun_dict = {
    "log": utils.t_log,
    "lin": utils.t_lin,
    "pow": utils.t_pow
}

n_fun_dict = {
    "a": n_arbitrary,
    "s": n_squares
}



def check_amount(arr):
    result = []

    xs = sample(range(9), 9)
    ys = sample(range(9), 9)
    indices = [(y, x) for y in ys for x in xs]
    np.random.shuffle(indices)

    for i in range(10, indices.__len__()-10):
        arr1 = np.array(arr, copy=True)

        for j in range(0, i):
            (y, x) = indices[j]
            arr1[y][x] = 0

        arr2 = fill(np.array(arr1, copy=True), arr1)
        result.append([i, utils.s_a2(arr2, arr1, 1000, utils.t_pow, n_squares, energy, plot_solution, utils.plot_xy, 10000, 1e-10, "results/sudoku/compare/" + str(i), 0)])

    utils.plot_xy(result, "results/sudoku/compare/chart")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    # parser.add_argument("--t0", type=float, required=True)
    # parser.add_argument("--t_fun", type=str, required=True)
    # parser.add_argument("--n_fun", type=str, required=True)
    # parser.add_argument("--max", type=int, required=True)
    # parser.add_argument("--eps", type=float, required=True)
    # parser.add_argument("--e_min", type=float, required=True)
    # parser.add_argument("--name", type=str, required=True)
    args = parser.parse_args()

    plt.gca().invert_yaxis()

    # Load matrix
    arr1 = load(args.input)

    check_amount(arr1)

    # # Fill empy fields
    # arr2 = fill(np.array(arr1, copy=True), arr1)
    #
    # e = utils.s_a2(arr2, arr1, args.t0, t_fun_dict[args.t_fun], n_fun_dict[args.n_fun], energy, plot_solution, utils.plot_xy, args.max, args.eps, args.name, args.e_min)
    # print(e)


