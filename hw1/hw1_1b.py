import numpy as np
import matplotlib.pyplot as plt


def risk(theta1, theta2):
    return theta1**2 + theta2**2 - 400*(theta1 + theta2)


minv, maxv = 0, 400
eval_nums = 400

theta1, theta2 = np.meshgrid(np.linspace(minv, maxv, eval_nums),
                             np.linspace(minv, maxv, eval_nums))

values = risk(theta1, theta2)

fig, ax = plt.subplots(figsize=(4, 4))
ax.grid(False)
CS = ax.contour(theta1, theta2, values, levels=10, linewidths=0.5)
ax.set_title("Contour for the Risk Function")
ax.set_xlabel(r"$\theta_1$ / m")
ax.set_ylabel(r"$\theta_2$ / m")
ax.clabel(CS)
fig.tight_layout()
fig.savefig("hw1_1b.pdf", dpi=500)
