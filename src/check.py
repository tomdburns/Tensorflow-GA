#!/usr/bin/env python

"""
Prints out a plot of the fitness function of the GA

Prints both best and average for each generation
"""


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sys import argv


FIT = False
if len(argv) > 0:
    for val in argv:
        if '-' in val:
            if 'r' in val:
                if not FIT:
                    FIT = True
                else:
                    FIT = False


def main():
    """Main Execution"""
    filename = 'Runningstats_GA.csv'
    stats = pd.read_csv(filename)
    gens = np.array(stats['Gen'])
    ngens = []
    for gen in gens:
        ngens.append(1 + gen)
    gens = np.array(ngens)
    best = np.array(stats['best'])
    mean = np.array(stats['mean'])
    ymax = max(mean) + 0.2 * max(mean)
    ymin = min(best) - 0.2 * min(best)

    if not FIT:
        new_best, new_mean = [], []
        for val in best:
            new_best.append(1 - val)
        for val in mean:
            new_mean.append(1 - val)
        best = np.array(new_best)
        mean = np.array(new_mean)
        ymax = max(best) + 0.4 * max(best)
        ymin = min(mean) - 0.2 * min(mean)

    print "Current Best: {}".format(max(best))

    plt.scatter(gens, mean, color='blue', label='Gen Mean')
    plt.scatter(gens, best, color='green', label='Gen Best')
    plt.title('GA Stats:')
    plt.xlabel('Generation')
    if FIT:
        plt.legend(loc='upper right')
        plt.ylabel('Fitness')
    else:
        plt.legend(loc='lower right')
        plt.ylabel('Pearson R Squared')
    plt.ylim(ymin, ymax)
    plt.xlim(0.0, max(gens) + 1)
    plt.show()

    new_best, nes_mean = [], []


if __name__ in '__main__':
    main()
