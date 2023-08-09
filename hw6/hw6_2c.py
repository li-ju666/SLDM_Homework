import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax

# sample size
ns = [10, 100, 1000]
action_prob = 0.3

key = jax.random.PRNGKey(0)

key, _ = jax.random.split(key)


# function to sample y condition on x and a from gaussian noise
def sample_y(x, a, noise):
    offset = jax.lax.cond(a, lambda _: x[0]*x[1], lambda _: 1-x[0]*x[1], None)
    return noise * jnp.sqrt(0.1) + offset


# vectorize the function
sample_ys = jax.vmap(sample_y, in_axes=(0, 0, 0))


# risk function
def risk(x, w, a, y):
    pax = jax.lax.cond((x[0]*x[1] < w) == a, lambda _: 1., lambda _: 0., None)
    pa = jax.lax.cond(a, lambda _: action_prob, lambda _: 1-action_prob, None)
    return y * pax / pa


# vectorize the function
risk_vec = jax.vmap(risk, in_axes=(0, None, 0, 0))

for n in ns:

    # generate random x
    xs = jax.random.uniform(key, shape=(n, 2)) * 2/3

    w_space = jnp.linspace(0, 1, 1000)

    risks = []
    for w in w_space:
        # sample actions
        key, _ = jax.random.split(key)
        actions = jax.random.bernoulli(key, p=action_prob, shape=(n,))

        # get random key to generate noise
        key, _ = jax.random.split(key)
        noises = jax.random.normal(key, shape=(n,))
        # sample ys
        ys = sample_ys(xs, actions, noises)

        # compute risk
        risks.append(risk_vec(xs, w, actions, ys).mean())

    # plot the results
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.grid(False)
    ax.plot(w_space, risks, alpha=0.5)

    ax.set_title(f"Risk estimated with randomized trial data\nSample size: {n}")
    ax.set_xlabel(r"$w$")
    ax.set_ylabel("Risk Estimation")

    # ax.legend()
    fig.savefig(f"hw6_2c_{n}.pdf", dpi=500)
    plt.close(fig)
