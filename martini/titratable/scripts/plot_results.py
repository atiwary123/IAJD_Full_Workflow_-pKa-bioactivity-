import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.optimize import curve_fit


def calc_degree_of_deprot(ph, pka, q):
    return 1/(10**(q*(pka-ph))+1)


parser = argparse.ArgumentParser(description='Plot the results')
parser.add_argument('-f', dest='file', type=str, default='results.txt', help='results file (.txt)')
parser.add_argument('-o', dest='out', type=str, default='results.pdf', help='output file (.pdf)')
args = parser.parse_args()

ph, degree_of_deprot, std = np.loadtxt(args.file, unpack=True)

fit = curve_fit(calc_degree_of_deprot, ph, degree_of_deprot, p0=(7.5, 0.75))[0]
ph_range = np.arange(3, 8.1, 0.1)

fig = plt.figure()

plt.plot(ph, degree_of_deprot, '^')
plt.plot(ph_range, calc_degree_of_deprot(ph_range, *fit))

plt.title(f'pKa = {fit[0]:.2f}, q = {fit[1]:.2f}')
plt.xlabel('pH')
plt.ylabel('Degree of deprotonation')

fig.tight_layout()
fig.savefig(args.out, bbox_inches='tight')
