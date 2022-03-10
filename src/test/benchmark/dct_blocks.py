from time import perf_counter
from modules.jpeg_pipeline.dct import *

import numpy as np
import matplotlib.pyplot as plt


def run(trials, min_side, max_side, block_size, step):
    times = list()
    times.append(list())
    times.append(list())
    times.append(list())
    times.append(list())
    times.append(list())
    times.append(list())

    c = int()

    for j in range(min_side, max_side, step):
        image = np.random.randint(255, size=(min_side, max_side))

        times[0].append(int())
        times[1].append(int())
        times[2].append(int())
        times[3].append(int())
        times[4].append(int())
        times[5].append(int())

        for k in range(trials):
            init_time = perf_counter()
            dct1 = apply_dct_blocks_loops(image, block_size)
            end_time = perf_counter()
            total = end_time - init_time
            times[0][c] += total

            init_time = perf_counter()
            dct2 = apply_dct_blocks_r_(image, block_size)
            end_time = perf_counter()
            total = end_time - init_time
            times[1][c] += total

            init_time = perf_counter()
            dct3 = apply_dct_blocks_optimized(image, block_size)
            end_time = perf_counter()
            total = end_time - init_time
            times[2][c] += total

            init_time = perf_counter()
            idct1 = apply_inverse_dct_blocks_loops(dct1, block_size)
            end_time = perf_counter()
            total = end_time - init_time
            times[3][c] += total

            init_time = perf_counter()
            idct2 = apply_inverse_dct_blocks_r_(dct2, block_size)
            end_time = perf_counter()
            total = end_time - init_time
            times[4][c] += total

            init_time = perf_counter()
            idct3 = apply_inverse_dct_blocks_optimized(dct3)
            end_time = perf_counter()
            total = end_time - init_time
            times[5][c] += total

            print("Side %d, Trial %d" % (j, k + 1))

        times[0][c] /= trials
        times[1][c] /= trials
        times[2][c] /= trials
        times[3][c] /= trials
        times[4][c] /= trials
        times[5][c] /= trials

        c += 1

    return times


def main():
    #trials = eval(input("Trials: "))
    #min_side = eval(input("Min side: "))
    #max_side = eval(input("Max side: "))
    #step = eval(input("Step: "))
    #block_size = eval(input("Block size: "))

    trials = 10
    min_side = 160
    max_side = 16 * 100
    step = 16
    block_size = 16

    times = run(trials, min_side, max_side, block_size, step)
    n = [i for i in range(min_side, max_side, step)]

    plt.figure()
    plt.title("Block DCT: range loop vs loop w/r_ vs numpy/astropy")
    plt.plot(n, times[0])
    plt.plot(n, times[1])
    plt.plot(n, times[2])
    plt.legend(["range loop", "loop w/r_", "numpy/astropy"])
    plt.show()

    plt.figure()
    plt.title("Block IDCT: range loop vs loop w/r_ vs numpy")
    plt.plot(n, times[3])
    plt.plot(n, times[4])
    plt.plot(n, times[5])
    plt.legend(["range loop", "loop w", "numpy"])
    plt.show()


if __name__ == '__main__':
    main()
