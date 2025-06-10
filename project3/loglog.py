import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, ScalarFormatter

filename = "times.txt"
sizes = []
times = []

with open(filename, 'r') as f:
    for l in f:
        if l.strip():
            N, t = map(float, l.split())
            sizes.append(N)
            times.append(t)

sizes = np.array(sizes)
times = np.array(times)

plt.figure(figsize=(8, 6))
plt.loglog(sizes, times, 'o-', base=10)
plt.xlabel("Problem size (N)")
plt.ylabel("Time elapsed (seconds)")
plt.title("Scaling Study (Log-Log Plot)")

ax = plt.gca()

ax.set_xticks(sizes)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

ax.yaxis.set_major_locator(LogLocator(base=10.0))
ax.yaxis.set_major_formatter(ScalarFormatter())

plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()