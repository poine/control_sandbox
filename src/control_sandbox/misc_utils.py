import math, numpy as np
"""
  Miscellanous...
"""
def get_om_xi(lambda1):
    om = math.sqrt(lambda1.real**2+lambda1.imag**2)
    xi = math.cos(np.arctan2(lambda1.imag, -lambda1.real))
    return om, xi

def get_lambdas(om, xi):
    re, im = -om*xi, om*math.sqrt(1-xi**2)
    return [complex(re, im), complex(re, -im)]

def get_precommand(A, B, C, K):
    tmp1 = np.linalg.inv(A - np.dot(B, K))
    tmp2 = np.dot(np.dot(C, tmp1), B)
    nr, nc = tmp2.shape
    H = -np.linalg.inv(tmp2) if nr == nc else -np.linalg.pinv(tmp2)
    return H

