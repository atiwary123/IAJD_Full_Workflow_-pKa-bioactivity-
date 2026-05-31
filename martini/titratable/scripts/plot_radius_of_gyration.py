import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Plot the results')
parser.add_argument('-f', dest='file', type=str, default='results_radius_of_gyration.txt',
                    help='results file (.txt)')
parser.add_argument('-o', dest='out', type=str, default='results_radius_of_gyration.pdf', 
                    help='output file (.pdf)')
args = parser.parse_args()

ph, radius_of_gyration, _ = np.loadtxt(args.file, unpack=True)

fig = plt.figure()
plt.plot(ph, radius_of_gyration, marker='D', linestyle='--')

plt.xlabel('pH')
plt.ylabel('Radius of gyration (nm)')

fig.tight_layout()
fig.savefig(args.out, bbox_inches='tight')
