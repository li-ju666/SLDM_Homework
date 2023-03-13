from scipy import stats
import numpy as np
from matplotlib import pyplot as plt

n = 1000


def s(x):
    return 1/(1+np.exp(-x))


a = stats.norm.rvs(0, 1, size=n)
x = np.array([stats.bernoulli.rvs(
    s(10*(each-0.5))) for each in a])
y = np.array([stats.norm.rvs(
    2*each, 1) for each in x])

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
ax1.scatter(a, y, s=2)
ax2.scatter(a[x.astype(bool)], y[x.astype(bool)],
            s=2)

unc_coref = np.corrcoef(a, y)[0, 1]
c_coref = np.corrcoef(a[x.astype(bool)], y[x.astype(bool)])[0, 1]

ax1.annotate(r"$\rho=$"+f"{unc_coref:.3f}", (-2.2, 4))
ax2.annotate(r"$\rho=$"+f"{c_coref:.3f}", (-2.2, 4))


ax1.set_xlabel(r"$a$")
ax1.set_ylabel(r"$y$")

ax2.set_xlabel(r"$a$")
ax2.set_ylabel(r"$y$")

ax1.set_xlim((-3, 3))
ax1.set_ylim((-3, 5))
ax2.set_xlim((-3, 3))
ax2.set_ylim((-3, 5))

ax1.set_title("Unconditional")
ax2.set_title(r"Conditional on $x=1$")

plt.tight_layout()
plt.savefig("hw4_2a.pdf", dpi=500)
