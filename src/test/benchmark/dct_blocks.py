"""------------DESTRUCTIVE COMPRESSION OF IMAGE - PARTIAL JPEG CODEC------------
University of Coimbra
Degree in Computer Science and Engineering
Multimedia
3rd year, 2nd semester
Authors:
Rui Bernardo Lopes Rodrigues, 2019217573, uc2019217573@student.uc.pt
Sancho Amaral Simões, 2019217590, uc2019217590@student.uc.pt
Tiago Filipe Santa Ventura, 2019243695, uc2019243695@student.uc.pt
Coimbra, 23rd March 2022
---------------------------------------------------------------------------"""

import matplotlib.pyplot as plt
from time import perf_counter
from modules.jpeg_pipeline.dct import *


def run(trials, min_side, max_side, block_size, step):
    """
    Function to run and calculate the time of the DCT
    :param trials: Number of times to run the test
    :param min_side: Minimum number in the random size
    :param max_side: Maximum number in the random size
    :param block_size: The block size
    :param step: for range step
    :return: the times of the trials.
    """
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
    """ Main function """
    #trials = eval(input("Trials: "))
    #min_side = eval(input("Min side: "))
    #max_side = eval(input("Max side: "))
    #step = eval(input("Step: "))
    #block_size = eval(input("Block size: "))

    trials = 5
    min_side = 160
    max_side = 16 * 400
    step = 16 * 4
    block_size = 8

    times = run(trials, min_side, max_side, block_size, step)
    n = [i for i in range(min_side, max_side, step)]

    plt.figure()
    plt.xlabel("Image side (pixels)")
    plt.ylabel("Execution time (s)")
    plt.title("DCT in blocks %dx%d: range loop vs r_ loop vs numpy/astropy" % (block_size, block_size))
    plt.plot(n, times[0])
    plt.plot(n, times[1])
    plt.plot(n, times[2])
    plt.legend(["range loop", "r_ loop", "numpy/astropy"])
    plt.show()

    plt.figure()
    plt.xlabel("Image side (pixels)")
    plt.ylabel("Execution time (s)")
    plt.title("IDCT in blocks %dx%d: range loop vs r_ loop vs numpy/astropy" % (block_size, block_size))
    plt.plot(n, times[3])
    plt.plot(n, times[4])
    plt.plot(n, times[5])
    plt.legend(["range loop", "r_ loop", "numpy/astropy"])
    plt.show()


if __name__ == '__main__':
    main()
