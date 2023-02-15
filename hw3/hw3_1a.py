import jax.numpy as jnp
import jax


thetas = [30, 40]
ns = [5, 50, 500]

K = 1000

key = jax.random.PRNGKey(42)


def _ln_factorial(z):
    return jax.lax.fori_loop(1, z+1, lambda z, acc: jnp.log(z)+acc, 0)


def loss(z, theta):
    return -(z*jnp.log(theta) - theta - _ln_factorial(z))


for n in ns:
    # generate true samples
    _, key = jax.random.split(key)
    true_zs = jax.random.poisson(key, 40, shape=(n,))

    # estimate theta
    theta_hat = jnp.average(true_zs)

    # function to calculate T
    vecloss = jax.vmap(lambda z: loss(z, theta_hat))
    T_fn = jax.jit(lambda zs: vecloss(zs).std()**2)

    T_true = T_fn(true_zs)

    for theta in thetas:
        less_than = 0
        for _ in range(K):
            _, key = jax.random.split(key)
            gen_zs = jax.random.poisson(key, theta, shape=(n,))
            T_gen = T_fn(gen_zs)
            less_than = less_than + 1 if T_gen <= T_true else less_than

        P_theta = less_than/K
        alpha_theta = min(P_theta, 1-P_theta)
        print(f"n={n}, theta={theta}: alpha_theta={alpha_theta:.3f}")
        # print(f"{n} & {theta} & {alpha_theta:.3f} & "
        #       f"{True if alpha_theta < 0.01 else False} \\\\")
