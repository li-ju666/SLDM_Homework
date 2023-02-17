import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt


key = jax.random.PRNGKey(42)

d = 100
sigma = 10
minval, maxval = 100, 170

keys = jax.random.split(key, num=d+1)
key, keys = keys[0], keys[1:d+1]
mus = jax.vmap(lambda key: jax.random.uniform(
    key, minval=minval, maxval=maxval))(keys)

keys = jax.random.split(key, num=d+1)
key, keys = keys[0], keys[1:d+1]
zs = jax.vmap(lambda key, m: jax.random.normal(key)*sigma+m, (0, 0))(keys, mus)

static_thetas = zs

hat_mu = zs.mean()
hat_v = jnp.maximum(((zs - hat_mu) ** 2).mean() - sigma**2, 0)

adapt_thetas = jax.vmap(lambda z: hat_v/(hat_v+sigma**2)*z +
                        sigma**2/(hat_v+sigma**2)*hat_mu)(zs)

print(static_thetas.shape, adapt_thetas.shape)

fig, ax = plt.subplots(figsize=(6, 4), nrows=2, ncols=1)

ax[0].bar(x=range(d), height=static_thetas, alpha=0.5)
ax[0].set_ylim((75, 200))
ax[0].set_title("Static Target Parameter")
ax[0].set_ylabel("Pressure / mmHg")

ax[1].bar(x=range(d), height=adapt_thetas, color='red', alpha=0.5)
ax[1].set_ylim((75, 200))
ax[1].set_title("Random Target Parameter")
ax[1].set_ylabel("Pressure / mmHg")
ax[1].set_xlabel("Patient ID")


# ax.set_title("Gradient Descent for optimization")
# ax.set_xlabel(r"$\theta_1$ / m")
# ax.set_ylabel(r"$\theta_2$ / m")

# ax.legend()
fig.tight_layout()
fig.savefig("hw3_4d.pdf", dpi=500)
