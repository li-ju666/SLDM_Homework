import jax.numpy as jnp
import numpy as np
import jax


ns = [5, 50, 500]
rs = [1e1, 1e4]

K = 1000

key = jax.random.PRNGKey(0)


def _ln_factorial(z):
    return jax.lax.fori_loop(1, z+1, lambda z, acc: jnp.log(z)+acc, 0)


def _stirling_ln_factorial(z):
    return z * jnp.log(z) - z + 1


def loss(z, theta):
    return -(z*jnp.log(theta) - theta - _ln_factorial(z))


def approx_loss(z, theta):
    return -(z*jnp.log(theta) - theta - _stirling_ln_factorial(z))


for n in ns:
    for r in rs:
        p = 40/(r+40)
        true_zs = np.random.negative_binomial(r, p, size=n)
        true_zs = jnp.array(true_zs)

        # estimate theta
        theta_hat = jnp.average(true_zs)

        # function to calculate T
        # Note: Stirling approximation is used for large z:
        # ln(z!) \approx nln(n)-n+1
        vecloss = jax.vmap(lambda z: loss(z, theta_hat)) if r <= 1e2 \
            else jax.vmap(lambda z: approx_loss(z, theta_hat))
        T_fn = jax.jit(lambda zs: vecloss(zs).std()**2)

        T_true = T_fn(true_zs)

        less_than = 0
        for _ in range(K):
            _, key = jax.random.split(key)
            gen_zs = jax.random.poisson(key, theta_hat, shape=(n,))
            T_gen = T_fn(gen_zs)
            less_than = less_than + 1 if T_gen <= T_true else less_than

        P_theta = less_than/K
        alpha_theta = min(P_theta, 1-P_theta)
        print(f"n={n}, r={r}, theta={theta_hat:.3f},"
              f"alpha_theta={alpha_theta:.3f}")
