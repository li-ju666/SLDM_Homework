import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax

# sample size
m = 1000

key = jax.random.PRNGKey(0)

key, _ = jax.random.split(key)

# generate random x
xs = jax.random.uniform(key, shape=(m, 2))


# policy function
def get_a(x, w):
    return x[0] * x[1] < w


# vectorize the policy function
get_as = jax.vmap(get_a, in_axes=(0, None))


# function to sample y condition on x and a from gaussian noise
def sample_y(x, a, noise):
    offset = jax.lax.cond(a, lambda _: x[0]*x[1], lambda _: 1-x[0]*x[1], None)
    return noise * jnp.sqrt(0.1) + offset


# vectorize the function
sample_ys = jax.vmap(sample_y, in_axes=(0, 0, 0))


w_space = jnp.linspace(0, 1, 1000)

risks = []
for w in w_space:
    # sample actions
    actions = get_as(xs, w)

    # get random key to generate noise
    key, _ = jax.random.split(key)

    noises = jax.random.normal(key, shape=(m,))
    # sample ys
    ys = sample_ys(xs, actions, noises)

    # compute risk
    risks.append(ys.mean())

# plot the results
fig, ax = plt.subplots(figsize=(4, 4))
ax.grid(False)
ax.plot(w_space, risks, alpha=0.5)

ax.set_title(r"Risk of policies with different values of $w$")
ax.set_xlabel(r"$w$")
ax.set_ylabel("Risk")

# ax.legend()
fig.savefig("hw6_1.pdf", dpi=500)
