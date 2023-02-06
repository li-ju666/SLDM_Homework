import numpy as np
import matplotlib.pyplot as plt

a1 = np.array([0, 0])
a2 = np.array([350, 50])
a3 = np.array([250, 350])
scirc = np.array([200, 200])
v = (1e-7)**2
c = 3e8
eps = 1e-7

n = 100

mean = 2/c*np.linalg.norm(scirc - np.array([a1, a2, a3]), axis=1)
cov = v * np.identity(3)

z = np.random.multivariate_normal(mean, cov, n)


def risk(z, s):
    mu = np.array([
        np.linalg.norm(s - a1),
        np.linalg.norm(s - a2),
        np.linalg.norm(s - a3),
    ])
    values = np.sum((z - 2/c*mu) ** 2, axis=1)
    return values.mean()


def grad(z, s):
    mu = np.array([
        np.linalg.norm(s - a1),
        np.linalg.norm(s - a2),
        np.linalg.norm(s - a3),
    ])
    # print(mu)
    gradient = 2*(2/c*mu - z)
    gradient = 2/c * gradient @ \
        np.array([(s - a1)/(mu[0] + eps),
                  (s - a2)/(mu[1] + eps),
                  (s - a3)/(mu[2] + eps)])
    gradient = np.mean(gradient, axis=0)
    return gradient


lr = 1e14
steps = 2000

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

s1, s2 = np.meshgrid(np.linspace(minv, maxv, eval_nums),
                     np.linspace(minv, maxv, eval_nums))

ss = np.array(list(zip(s1.flatten(), s2.flatten())))

values = np.array(list(map(lambda s: risk(z, s), ss))).reshape((eval_nums, -1))

fig, ax = plt.subplots(figsize=(4, 4))
ax.grid(False)
CS = ax.contour(s1, s2, values, levels=10, linewidths=0.5)
ax.plot(thetas[:-1, 0], thetas[:-1, 1], '-o',
        markersize=3, linewidth=0.5, label="GD steps")
opt_point = thetas[-1]
ax.annotate(f'({opt_point[0]:.3f},{opt_point[1]:.3f})', xy=opt_point+5)
ax.plot(opt_point[0], opt_point[1], 'ro', alpha=0.5, label="Minimal")
ax.set_title("Gradient Descent for optimization")
ax.set_xlabel(r"$\theta_1$ / m")
ax.set_ylabel(r"$\theta_2$ / m")
ax.clabel(CS)
ax.legend()
fig.tight_layout()
fig.savefig("hw2_4b.pdf", dpi=500)
