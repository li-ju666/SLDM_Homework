import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax

# sample size
m = 1000

key = jax.random.PRNGKey(0)

key, _ = jax.random.split(key)

# generate random x
xs = jax.random.uniform(key, shape=(m,))

# policy function
policy = jax.vmap(lambda x, w: 0 if jnp.multiply(x, jnp.ones(2)) >= w else 1,
                  in_axes=(0, None))


# function to sample y condition on x and a
def sample_y(x, a, key):
    key, _ = jax.random.split(key)
    y = jax.random.normal(key, 1-x[0]*x[1], 0.1) if a else jax.random.normal(key, x[0]*x[1], 0.1)
    return y


# vectorize the function
sample_ys = jax.vmap(sample_y, in_axes=(0, 0, None))


# w_space = jnp.linspace(0, 1, 1000)
w_space = [0.5]

risks = []
for w in w_space:
    # compute actions
    key = jax.random.split(key)
    actions = policy(xs, w)
    ys = sample_ys(xs, actions, key)

    # compute risk
    risks.append(ys.sum())

# plot the results

fig, ax = plt.subplots(figsize=(4, 4))
ax.grid(False)
ax.plot(w_space, risks, s=2, c="blue", alpha=0.5)

ax.set_title(r"Risk of policies with different values of $w$")
ax.set_xlabel(r"$w$")
ax.set_ylabel("Risk")

ax.legend()
fig.savefig("hw6_1.pdf", dpi=500)
