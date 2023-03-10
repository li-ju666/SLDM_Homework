import numpy as np
from scipy import stats
import cvxpy as cp


n = 100
eps = 0.2
eps_tilde = 0.4

# parameters for true distributions
mu = np.array([200, 200])
sigma = np.array([[400, 50], [50, 400]])
v = 2.5
c = 2 * v / (v - 2) * sigma


# generate samples from true mixed distributions
choices = stats.bernoulli.rvs(1-eps, size=n).astype(bool)

uncorrupted = stats.multivariate_normal.rvs(
    mu, sigma, size=n,
)

corrupts = stats.multivariate_t.rvs(
    mu, c, df=v, size=n,
)
zs = np.concatenate([
    uncorrupted[choices],
    corrupts[np.logical_not(choices)],
])
np.random.shuffle(zs)
# zs = jnp.array(zs)


def loss(zs, theta):
    return ((zs - theta) ** 2).sum(axis=1)


def estimate_theta(weights, zs):
    return np.average(zs, axis=0, weights=weights)


def estimate_weights(zs, theta, eps_tilde):
    losses = loss(zs, theta).transpose()
    weights = cp.Variable(n, pos=True)
    prob = cp.Problem(
        cp.Minimize(losses @ weights),
        [np.ones(n).transpose() @ weights == 1,
         cp.sum(cp.entr(weights)) >=
         np.log((1-eps_tilde)*zs.shape[0]),
         ])
    # print(prob.is_dcp())
    # prob.solve(verbose=True)
    return weights.value


weights = 1/n*np.ones(n)

for i in range(100):
    theta = estimate_theta(weights, zs)
    weights = estimate_weights(zs, theta, eps_tilde)
    print(f"Step {i}: {theta} ===============")
