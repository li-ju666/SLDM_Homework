import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
from sklearn.linear_model import LogisticRegression


m = 1000

# initialize mu0, sigma0, mu1, sigma1
mu0 = jnp.array([50, 140])
sigma0 = jnp.array([[64, 9], [9, 64]])

mu1 = jnp.array([60, 160])
sigma1 = jnp.array([[64, 49], [49, 64]])

# initialize key for random number generation
key = jax.random.PRNGKey(0)

# generate samples
key, _ = jax.random.split(key)
y0_nums = jax.random.bernoulli(key, p=0.2, shape=(m,)).sum().item()

key, _ = jax.random.split(key)
x0s = jax.random.multivariate_normal(key, mu0, sigma0, shape=(y0_nums,))

key, _ = jax.random.split(key)
x1s = jax.random.multivariate_normal(
    key, mu1, sigma1, shape=(m-y0_nums,))


# define transformation functions
phi1 = jax.vmap(lambda x: jnp.array([1, x[0], x[1]]),
                in_axes=(0,))

phi2 = jax.vmap(lambda x: jnp.array([1, x[0], x[1],
                                     x[0]**2, x[1]**2,
                                     x[0]*x[1]]),
                in_axes=(0,))

xs = jnp.concatenate((x0s, x1s))
ys = jnp.concatenate((jnp.zeros(y0_nums), jnp.ones(m-y0_nums)))


# compute false positive rate and false negative rate
def fp_fn_compute(tau, model, xs, ys):
    # compute log odds
    log_odds = model.intercept_ + jnp.dot(model.coef_, xs.T)
    # compute predictions
    predictions = (log_odds > tau).astype(int)
    # compute L0
    fp_rate = jnp.sum((predictions == 1) & (ys == 0)) / jnp.sum(ys == 0)
    # compute 1-L1
    fn_rate = jnp.sum((predictions == 1) & (ys == 1)) / jnp.sum(ys == 1)

    return fp_rate, fn_rate


nums_samples = [10, 100, 1000]
phis = {"phi1": phi1, "phi2": phi2}
for phi_name, phi in phis.items():
    # define logistic regression model
    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(phi(xs), ys)

    # sample n points from xs
    for n in nums_samples:
        key, _ = jax.random.split(key)
        sample_indices = jax.random.choice(key, jnp.arange(m), shape=(n,),
                                           replace=False)
        sample_xs = phi(xs[sample_indices])
        sample_ys = ys[sample_indices]

        fps = []
        fns = []
        tau_range = jnp.linspace(-50, 50, 1000)
        for tau in tau_range:
            fp_rate, fn_rate = fp_fn_compute(tau, model,
                                             sample_xs, sample_ys)
            fps.append(fp_rate)
            fns.append(fn_rate)

        # print the results
        plt.plot(fps, fns)
        plt.xlabel(r'$L_0$')
        plt.ylabel(r'$1-L_1$')
        title = f'FPR vs TPR: {n} samples with {phi_name}'
        plt.title(title)
        plt.savefig(f'hw5/figure2b/{title.replace(" ", "").replace(":", "")}.pdf', dpi=300)
        plt.close()
