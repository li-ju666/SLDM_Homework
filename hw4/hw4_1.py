import numpy as np
from scipy import stats
import cvxpy as cp
# from matplotlib import pyplot


n = 100
eps = 0.2
eps_tildes = [0.1, 0.2, 0.3, 0.4]

# parameters for true distributions
mu = np.array([200, 200])
sigma = np.array([[400, 50], [50, 400]])
v = 2.5
c = 2 * v / (v - 2) * sigma


# generate samples from true mixed distributions
corrupted_nums = stats.bernoulli.rvs(eps, size=n).sum()

uncorrupted = stats.multivariate_normal.rvs(
    mu, sigma, size=n-corrupted_nums,
)

corrupts = stats.multivariate_t.rvs(
    mu, c, df=v, size=corrupted_nums,
)

zs = np.concatenate([uncorrupted, corrupts])
np.random.shuffle(zs)

# pyplot.scatter(x=uncorrupted[:, 0],
#                y=uncorrupted[:, 1],
#                c="blue")
# pyplot.scatter(x=corrupts[:, 0],
#                y=corrupts[:, 1],
#                c="red")

# pyplot.show()


def loss(zs, theta):
    return ((zs - theta) ** 2).sum(axis=1)


def estimate_theta(weights, zs):
    return np.average(zs, axis=0, weights=weights)


def estimate_weights(zs, theta, eps_tilde):
    losses = loss(zs, theta).transpose()
    # print(losses)
    weights = cp.Variable(n, pos=True)
    prob = cp.Problem(
        cp.Minimize(losses @ weights),
        [np.ones(n).transpose() @ weights == 1,
         cp.sum(cp.entr(weights)) >=
         np.log((1-eps_tilde)*zs.shape[0]),
         ])
    prob.solve(verbose=False)
    # prob.solve(verbose=True)
    return weights.value


for eps_tilde in eps_tildes:

    weights = 1/n*np.ones(n)
    last_theta = np.array([0., 0.])
    threshold = 1e-4
    improvement = 1e3

    while improvement >= threshold:
        theta = estimate_theta(weights, zs)
        improvement = np.linalg.norm(theta - last_theta)
        last_theta = theta
        weights = estimate_weights(zs, theta, eps_tilde)
    print(f"Eps_tilde: {eps_tilde} with theta: {theta}")
print(f"ERM theta {estimate_theta(None, zs)}")
