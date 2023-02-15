import jax.numpy as jnp
import jax
import numpy as np


# settings
ns = [10, 100, 1000]

c = 3e8
a1 = jnp.array([0, 0])
a2 = jnp.array([350, 50])
a3 = jnp.array([250, 350])
s = jnp.array([200, 200])
eps = 1e-10
d = 3
K = 1000

key = jax.random.PRNGKey(42)


# functions for sn optimization: single sample
def _sn_objective_sample(z, s):
    mu = jnp.array([
        jnp.linalg.norm(s - a1),
        jnp.linalg.norm(s - a2),
        jnp.linalg.norm(s - a3),
    ])
    values = (z - 2/c*mu) ** 2
    return values.sum()


# vectorized objective function
sn_objective_vec = jax.jit(jax.vmap(_sn_objective_sample, (0, None)))


# gradient of the sn objective function: single sample
def _sn_grad_sample(z, s):
    mu = jnp.array([
        jnp.linalg.norm(s - a1),
        jnp.linalg.norm(s - a2),
        jnp.linalg.norm(s - a3),
    ])
    gradient = 2*(2/c*mu - z)
    gradient = 2/c * gradient @ \
        jnp.array([(s - a1)/(mu[0] + eps),
                   (s - a2)/(mu[1] + eps),
                   (s - a3)/(mu[2] + eps)])
    return gradient


# vectorized grad for sn objective function
sn_grad = jax.jit(lambda zs, s: jax.vmap(_sn_grad_sample,
                                         (0, None))(zs, s).mean(axis=0))


# loss function: single sample
def loss_sample(z, s, v):
    return d/2 + jnp.log(v) + _sn_objective_sample(z, s)/(2*v)


# GD to estimate sn
def get_sn(start, num_steps, lr, zs):
    for _ in range(num_steps):
        gradient = sn_grad(zs, start)
        start -= lr * gradient
    return start


true_data_generators = {
    "gaussian": lambda key, mean, cov, num:
        jax.random.multivariate_normal(key, mean, cov, shape=(num, )),
    "exp": lambda key, mean, cov, num:
        jnp.array([np.random.exponential(mean[i], num)
                   for i in range(3)]).transpose()
}


mean = 2/c*jnp.linalg.norm(s - jnp.array([a1, a2, a3]), axis=1)
cov = 1e-14 * jnp.eye(3)

for generator_name in true_data_generators.keys():
    generator = true_data_generators[generator_name]
    for n in ns:
        # estimate thetas
        _, key = jax.random.split(key)
        true_zs = generator(key, mean, cov, n)

        # modeling
        sn = get_sn(start=jnp.array([0.0, 0.0]),
                    num_steps=2000,
                    lr=1e15,
                    zs=true_zs)
        vn = sn_objective_vec(true_zs, sn).mean() / 3

        loss_vec = jax.vmap(lambda z: loss_sample(z, sn, vn))
        T_fn = jax.jit(lambda zs: loss_vec(zs).std() ** 2)

        T_true = T_fn(true_zs)

        less_than = 0
        est_mean = 2/c*jnp.linalg.norm(sn - jnp.array([a1, a2, a3]), axis=1)
        est_cov = vn*jnp.eye(3)
        for _ in range(K):
            _, key = jax.random.split(key)
            gen_zs = jax.random.multivariate_normal(key, est_mean,
                                                    est_cov, shape=(n, ))
            T_gen = T_fn(gen_zs)
            less_than = less_than + 1 if T_gen <= T_true else less_than

        P_theta = less_than/K
        alpha_theta = min(P_theta, 1-P_theta)
        print(f"{generator_name} & {n} & {sn} & {vn} & "
              f"{alpha_theta:.3f} & "
              f"{True if alpha_theta < 0.01 else False} \\\\")
