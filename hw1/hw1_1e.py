import numpy as np
import matplotlib.pyplot as plt


mean = [200, 200]
cov = [[400, 50], [50, 400]]
ns = [1, 10, 100, 1000]

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))

for idx, n in enumerate(ns):
    row, col = idx // 2, idx % 2
    ax = axs[row][col]
    z = np.random.multivariate_normal(mean, cov, n)

    def risk(z, theta1, theta2):
        # shape of z: num * 2
        t1 = np.sum(z * z) / z.shape[0]
        avg = np.sum(z, 0)/z.shape[0]
        t2 = -2 * ((theta1 * avg[0]) + (theta2 * avg[1]))
        t3 = theta1 ** 2 + theta2 ** 2
        return t1 + t2 + t3

    minv, maxv = 0, 400
    eval_nums = 400

    theta1, theta2 = np.meshgrid(np.linspace(minv, maxv, eval_nums),
                                 np.linspace(minv, maxv, eval_nums))

    values = risk(z, theta1, theta2)

    ax.grid(False)
    CS = ax.contour(theta1, theta2, values, levels=10, linewidths=0.5)
    opt_point = np.sum(z, 0)/z.shape[0]
    ax.annotate(f'{opt_point[0]:9.2f},{opt_point[1]:9.2f}', xy=opt_point+5)
    ax.plot(opt_point[0], opt_point[1], 'ro', alpha=0.5, label="Mininal")
    ax.set_title(f"Risk Function: n={n}")
    ax.set_xlabel(r"$\theta_1$ / m")
    ax.set_ylabel(r"$\theta_2$ / m")
    ax.clabel(CS)
    ax.legend()
fig.tight_layout()
fig.savefig("hw1_1e.pdf", dpi=500)
