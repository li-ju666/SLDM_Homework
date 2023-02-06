import numpy as np
import scipy.stats as st


a = np.array([50, 100])
ns = [100, 1000]
conf_levs = [0.9, 0.95, 0.99]

mu_z = [200, 200]
cov_z = [[400, 50], [50, 400]]

for n in ns:
    # sampling
    zs = np.random.multivariate_normal(mu_z, cov_z, n)
    # the point estimation of theta
    theta_n = np.sum(zs, 0)/zs.shape[0]
    # the variance of $\sqrt{n}(\hat{\theta} - \theta_\circ)$
    v_theta = (zs - theta_n).transpose() @ (zs - theta_n) / zs.shape[0]
    # the variance of $\sqrt{n}(\hat{\tau} - \tau_\circ)$
    v_n = 1/np.linalg.norm(theta_n - a) ** 2 * \
        (theta_n - a) @ v_theta @ (theta_n - a).transpose()
    for conf_lev in conf_levs:
        # calculate confidence interval with normal distribution
        intv = st.norm.interval(confidence=conf_lev,
                                loc=np.linalg.norm(theta_n - a),
                                scale=np.sqrt(v_n/n))
        print(f"{n} samples with confidence level {conf_lev}: "
              f"({intv[0]:.3f}, {intv[1]: .3f})")
