import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax

# sample size
ns = [100, 1000]

key = jax.random.PRNGKey(0)

key, _ = jax.random.split(key)


def get_action(x, key):
    prob = 1 - jax.nn.sigmoid(x[0] * x[1] + 1)
    return jax.random.bernoulli(key, p=prob)


get_actions = jax.vmap(get_action, in_axes=(0, 0))


# function to sample y condition on x and a from gaussian noise
def sample_y(x, a, noise):
    offset = jax.lax.cond(a, lambda _: x[0]*x[1], lambda _: 1-x[0]*x[1], None)
    return noise * jnp.sqrt(0.1) + offset


# vectorize the function
sample_ys = jax.vmap(sample_y, in_axes=(0, 0, 0))


# risk function
def risk(x, w, a, y):
    pax = jax.lax.cond((x[0]*x[1] < w) == a, lambda _: 1., lambda _: 0., None)
    a0_prob = jax.nn.sigmoid(x[0] * x[1] + 1)
    tilde_pax = jax.lax.cond(a, lambda _: 1-a0_prob, lambda _: a0_prob, None)
    return y * pax / tilde_pax


# vectorize the function
risk_vec = jax.vmap(risk, in_axes=(0, None, 0, 0))

for n in ns:

    # generate random x
    xs = jax.random.uniform(key, shape=(n, 2))

    w_space = jnp.linspace(0, 1, 1000)

    risks = []
    for w in w_space:
        # sample actions
        keys = jax.random.split(key, n+1)
        key = keys[-1]
        actions = get_actions(xs, keys[:n])

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

    ax.set_title(f"Risk estimated with logistic policy\nSample size: {n}")
    ax.set_xlabel(r"$w$")
    ax.set_ylabel("Risk Estimation")

    # ax.legend()
    fig.savefig(f"hw6_3_{n}.pdf", dpi=500)
    plt.close(fig)
