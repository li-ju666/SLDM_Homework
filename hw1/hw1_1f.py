import numpy as np
import matplotlib.pyplot as plt


mean = [200, 200]
cov = [[400, 50], [50, 400]]
n = 1000

z = np.random.multivariate_normal(mean, cov, n)


def risk(z, theta1, theta2):
    t1 = np.sum(z * z) / z.shape[0]
    avg = np.sum(z, 0)/z.shape[0]
    t2 = -2 * ((theta1 * avg[0]) + (theta2 * avg[1]))
    t3 = theta1 ** 2 + theta2 ** 2
    return t1 + t2 + t3


def grad(z, theta):
    avg = np.sum(z, 0)/z.shape[0]
    return 2*(theta - avg)


lr = 0.01
steps = 500

theta = np.array([0, 0])
thetas = []
for i in range(steps):
    if not i % 10:
        thetas.append(theta)
    gradient = grad(z, theta)
    theta = theta - lr * gradient

thetas = np.array(thetas)
print(thetas)
# plot GD steps
minv, maxv = 0, 400
eval_nums = 400

theta1, theta2 = np.meshgrid(np.linspace(minv, maxv, eval_nums),
                             np.linspace(minv, maxv, eval_nums))

values = risk(z, theta1, theta2)

fig, ax = plt.subplots(figsize=(4, 4))
ax.grid(False)
CS = ax.contour(theta1, theta2, values, levels=10, linewidths=0.5)
ax.plot(thetas[:-1, 0], thetas[:-1, 1], '-o',
        markersize=3, linewidth=0.5, label="GD steps")
opt_point = np.sum(z, 0)/z.shape[0]
ax.annotate(f'{opt_point[0]:9.2f},{opt_point[1]:9.2f}', xy=opt_point+5)
ax.plot(opt_point[0], opt_point[1], 'ro', alpha=0.5, label="Mininal")
ax.set_title("Gradient Descent for optimization")
ax.set_xlabel(r"$\theta_1$ / m")
ax.set_ylabel(r"$\theta_2$ / m")
ax.clabel(CS)
ax.legend()
fig.tight_layout()
fig.savefig("hw1_1f.pdf", dpi=500)
