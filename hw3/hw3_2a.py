import numpy as np
from numpy.random import multivariate_normal, exponential

ns = [10, 100, 1000, 10000]

c = 3e8
a1 = np.array([0, 0])
a2 = np.array([350, 50])
a3 = np.array([250, 350])
s = np.array([200, 200])
eps = 1e-7
K = 1000


mean = 2/c*np.linalg.norm(s - np.array([a1, a2, a3]), axis=1)
cov = 1e-14 * np.identity(3)


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


def get_sn(start, num_steps, lr, z):
    for _ in range(num_steps):
        gradient = grad(z, start)
        start -= lr * gradient
    return start


def consistent_test(true_zs, sn, vn, K):
    less_than = 0
    n = true_zs.shape[0]
    mean = 2/c*np.linalg.norm(sn - np.array([a1, a2, a3]), axis=1)
    cov = vn * np.identity(3)
    for _ in range(K):
        gen_zs = multivariate_normal(mean, cov, size=n)
        E_gen = np.average(gen_zs)
        T_true = np.average(np.square(true_zs - E_gen))
        T_gen = np.average(np.square(gen_zs - E_gen))
        less_than = less_than + 1 if T_gen <= T_true else less_than
    return less_than/K


for n in ns:
    # estimate thetas
    true_zs = multivariate_normal(mean, cov, size=n)
    sn = get_sn(start=np.array([0.0, 0.0]),
                num_steps=2000,
                lr=1e14,
                z=true_zs)
    vn = risk(true_zs, sn) / 3
    print(f"sn: {sn}, vn: {vn}")
    # generate data
    P_theta = consistent_test(true_zs, sn, vn, K)
    alpha_theta = min(P_theta, 1-P_theta)
    print(f"Gaussian data: n={n}, alpha_theta={alpha_theta:.3f}")
print("===========")

for n in ns:
    # estimate thetas
    true_zs = list(map(lambda m: exponential(m, size=n), mean))
    true_zs = np.array(true_zs).transpose()
    sn = get_sn(start=np.array([0.0, 0.0]),
                num_steps=2000,
                lr=1e14,
                z=true_zs)
    vn = risk(true_zs, sn) / 3
    print(f"sn: {sn}, vn: {vn}")
    # generate data
    P_theta = consistent_test(true_zs, sn, vn, K)
    alpha_theta = min(P_theta, 1-P_theta)
    print(f"Exponential data: n={n}, alpha_theta={alpha_theta:.3f}")
