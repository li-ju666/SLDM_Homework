import jax.numpy as jnp
import jax


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
                                         (0, None))(zs, s).mean())


# loss function: single sample
def loss_sample(z, s, v):
    return d/2 + jax.log(v) + _sn_objective_sample(z, s)/(2*v)


# vectorized loss function
loss_vec = jax.jit(jax.vmap(loss_sample, (0, None, None), 0))


def get_sn(start, num_steps, lr, zs):
    for _ in range(num_steps):
        gradient = sn_grad(zs, start)
        start -= lr * gradient
    return start


mean = 2/c*jnp.linalg.norm(s - jnp.array([a1, a2, a3]), axis=1)
cov = 1e-14 * jnp.eye(3)

for n in ns:
    # estimate thetas
    _, key = jax.random.split(key)
    true_zs = jax.random.multivariate_normal(key, mean, cov, shape=(n, ))

    sn = get_sn(start=jnp.array([0.0, 0.0]),
                num_steps=2000,
                lr=1e15,
                zs=true_zs)
    vn = sn_objective_vec(true_zs, sn).mean() / 3
    print(f"sn: {sn}, vn: {vn}")
print("===========")
