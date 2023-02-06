from numpy.random import poisson
import numpy as np
import scipy.stats as st

n = 200
m = n//2

lambda0 = 100
lambda1 = 120

s0_samples = poisson(lambda0, m)
s1_samples = poisson(lambda1, m)

mean0 = np.mean(s0_samples)
mean1 = np.mean(s1_samples)
tau_hat = mean1 - mean0

# inverse of Qn
Qninv = np.array([[n*(mean0**2)/np.sum(s0_samples), 0],
               [0, n*(mean1**2)/np.sum(s1_samples)]])

# Ln
Ln = np.array([[np.sum((1-s0_samples/mean0)**2)/n, 0],
               [0, np.sum((1-s1_samples/mean1)**2)/n]])

# tau_dot
taudot = np.array([-1, 1])

# variance of \sqrt{n}\hat{\tau}
v_n = taudot.transpose() @ Qninv @ Ln @ Qninv @ taudot

conf_levs = [0.9, 0.95, 0.99]
for conf_lev in conf_levs:
    intv = st.norm.interval(confidence=conf_lev,
                            loc=np.linalg.norm(tau_hat),
                            scale=np.sqrt(v_n/n))
    print(f"[{intv[0]: .3f}, {intv[1]: .3f}]"
          f" with confidence level {conf_lev}")
