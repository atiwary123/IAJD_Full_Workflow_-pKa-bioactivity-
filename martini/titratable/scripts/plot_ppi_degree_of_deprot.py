import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def calc_degree_of_deprot(ph, pka, q):
    return 1/(10**(q*(pka-ph))+1)


ph_range = np.arange(3, 8.1, 0.1)
fig = plt.figure()


for i in range(1, 6):
    ph, degree_of_deprot, std = np.loadtxt(f'results_G{i}.txt', unpack=True)
    fit = curve_fit(calc_degree_of_deprot, ph, degree_of_deprot, p0=(7.5, 0.75))[0]

    points, = plt.plot(ph, degree_of_deprot, '^',
                      label=f'G{i}, pKa = {fit[0]:.2f}, q = {fit[1]:.2f}')
    plt.plot(ph_range, calc_degree_of_deprot(ph_range, *fit), color=points.get_color())

plt.xlabel('pH')
plt.ylabel('Degree of deprotonation')
plt.legend()
fig.tight_layout()
fig.savefig('results.pdf', bbox_inches='tight')

