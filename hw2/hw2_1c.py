import numpy as np
import scipy.stats as st


a = np.array([50, 100])
ns = [100, 1000]
conf_lev = 0.95

mu_z = [200, 200]
cov_z = [[400, 50], [50, 400]]

tau_o = np.linalg.norm(np.array(mu_z) - a)
M = int(1e4)
for n in ns:
    covered = 0
    for _ in range(M):
        # sampling
        zs = np.random.multivariate_normal(mu_z, cov_z, n)
        # the point estimation of theta
        theta_n = np.sum(zs, 0)/zs.shape[0]
        # the variance of $\sqrt{n}(\hat{\theta} - \theta_\circ)$
        v_theta = (zs - theta_n).transpose() @ (zs - theta_n) / zs.shape[0]
        # the variance of $\sqrt{n}(\hat{\tau} - \tau_\circ)$
        v_n = 1/np.linalg.norm(theta_n - a) ** 2 * \
            (theta_n - a) @ v_theta @ (theta_n - a).transpose()
        # calculate confidence interval with normal distribution
        intv = st.norm.interval(confidence=conf_lev,
                                loc=np.linalg.norm(theta_n - a),
                                scale=np.sqrt(v_n/n))
        if tau_o >= intv[0] and tau_o <= intv[1]:
            covered += 1
        # else:
        #     print("NOT IN!")
    print(f"Sample size {n}: Coverage: {covered/M}")