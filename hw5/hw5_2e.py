import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax

# generate data
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

xs = jnp.concatenate((x0s, x1s))
ys = jnp.concatenate((jnp.zeros(y0_nums), jnp.ones(m-y0_nums)))


# define predict function
def predict(x, tau, mu0_hat, mu1_hat, sigma0_hat, sigma1_hat):
    # compute ratio of two likelihoods
    py1 = jax.scipy.stats.multivariate_normal.pdf(x, mu1_hat, sigma1_hat)
    py0 = jax.scipy.stats.multivariate_normal.pdf(x, mu0_hat, sigma0_hat)
    predictions = (jnp.log(py1/py0) > tau).astype(int)
    return predictions


# compute L0 and 1-L1
def fp_fn_compute(predictions, ys):
    # compute L0
    fp_rate = jnp.sum((predictions == 1) & (ys == 0)) / jnp.sum(ys == 0)
    # compute 1-L1
    fn_rate = jnp.sum((predictions == 1) & (ys == 1)) / jnp.sum(ys == 1)

    return fp_rate, fn_rate


# estimate mean values
mu0_hat = jnp.mean(x0s, axis=0)
mu1_hat = jnp.mean(x1s, axis=0)

# estimate the covariance matrix when assuming the same covariance matrix
sigma_hat = jnp.dot((x0s - mu0_hat).T, (x0s - mu0_hat))/m + \
    jnp.dot((x1s - mu1_hat).T, (x1s - mu1_hat))/m

# estimate the covariance matrix when assuming different covariance matrices
sigma0_hat = jnp.dot((x0s - mu0_hat).T, (x0s - mu0_hat))/y0_nums
sigma1_hat = jnp.dot((x1s - mu1_hat).T, (x1s - mu1_hat))/(m-y0_nums)

sigmas = {"Linear": (sigma_hat, sigma_hat),
          "Quadratic": (sigma0_hat, sigma1_hat)}

# print all estimated values
print(f'mu0_hat: {mu0_hat}')
print(f'mu1_hat: {mu1_hat}')
print(f'sigma_hat: {sigma_hat}')
print(f'sigma0_hat: {sigma0_hat}')
print(f'sigma1_hat: {sigma1_hat}')

tau_range = jnp.linspace(-50, 50, 1000)
nums_samples = [10, 100, 1000]


for model_classs, sigma_values in sigmas.items():
    for n in nums_samples:
        key, _ = jax.random.split(key)
        sample_indices = jax.random.choice(key, jnp.arange(m), shape=(n,),
                                           replace=False)
        sample_xs = xs[sample_indices]
        sample_ys = ys[sample_indices]

        fps = []
        fns = []
        for tau in tau_range:
            predictions = predict(sample_xs, tau, mu0_hat, mu1_hat,
                                  sigma_values[0], sigma_values[1])
            fp_rate, fn_rate = fp_fn_compute(predictions, sample_ys)
            fps.append(fp_rate)
            fns.append(fn_rate)

        # print the results
        plt.plot(fps, fns)
        plt.xlabel(r'$L_0$')
        plt.ylabel(r'$1-L_1$')
        title = f'FPR vs TPR: {n} samples ({model_classs} model)'
        plt.title(title)
        plt.savefig(f'hw5/figure2e/{title.replace(" ", "").replace(":", "")}.pdf', dpi=300)
        plt.close()
