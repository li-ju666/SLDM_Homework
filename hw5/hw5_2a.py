import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax

m = 1000

mu0 = jnp.array([50, 140])
sigma0 = jnp.array([[64, 9], [9, 64]])

mu1 = jnp.array([60, 160])
sigma1 = jnp.array([[64, 49], [49, 64]])

key = jax.random.PRNGKey(0)

key, _ = jax.random.split(key)
y0_nums = jax.random.bernoulli(key, p=0.2, shape=(m,)).sum().item()

key, _ = jax.random.split(key)
x0s = jax.random.multivariate_normal(key, mu0, sigma0, shape=(y0_nums,))

key, _ = jax.random.split(key)
x1s = jax.random.multivariate_normal(
    key, mu1, sigma1, shape=(m-y0_nums,))

fig, ax = plt.subplots(figsize=(4, 4))
ax.grid(False)
ax.scatter(x0s[:, 0], x0s[:, 1], s=2,
           label="Healthy", c="blue", alpha=0.5)
ax.scatter(x1s[:, 0], x1s[:, 1], s=2,
           label="Ill", c="red", alpha=0.5)

ax.set_title("Covariate samples of healthy and ill patients")
ax.set_xlabel("Age")
ax.set_ylabel("LDL cholesterol level")

ax.legend()
fig.tight_layout()
fig.savefig("hw5_2a.pdf", dpi=500)
