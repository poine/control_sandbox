import os, math, numpy as np
import pickle

"""
  Miscellanous...
"""
# we assume _this_ file is cs_dir/src/control_sandbox/misc_utils.py
def cs_dir():
    dirname, filename = os.path.split(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(dirname, '../..'))

def cs_asset(asset): return os.path.join(cs_dir(), asset)


def save_trajectory(time, X, U, desc, filename):
    with open(filename, "wb") as f:
        pickle.dump([time, X, U, desc], f)

def load_trajectory(filename):
    with open(filename, "rb") as f:
        time, X, U, desc = pickle.load(f)
    return time, X, U, desc



"""
Misc
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


class CtlNone:
    def __init__(self, yc=None):
        self.yc = yc

    def get(self, X, k):
        return self.yc[k]

class IoCtlCst:
    def __init__(self, ysp):
        self.ysp = ysp

    def get(self, k, y_k, y_km1, u_km1):
        return self.ysp[k]

def make_random_pulses(dt, size, min_nperiod=1, max_nperiod=10, min_intensity=-1, max_intensity=1.):
    ''' make a vector of pulses of randon duration and intensities '''
    npulses = int(size/max_nperiod*2)
    durations = np.random.random_integers(low=min_nperiod, high=max_nperiod, size=npulses)
    intensities =  np.random.uniform(low=min_intensity, high=max_intensity, size=npulses)
    pulses = []
    for duration, intensitie in zip(durations, intensities):
        pulses += [intensitie for i in range(duration)]
    pulses = np.array(pulses)
    time = np.linspace(0, dt*len(pulses), len(pulses))
    return time, pulses

def step(t, a0=-1, a1=1, dt=4, t0=0): return a0 if math.fmod(t+t0, dt) > dt/2 else a1
def step_input_vec(time, a0=-1, a1=1, dt=4, t0=0): return [step(t, a0, a1, dt, t0) for t in time]
def step_vec(time, a0=-1, a1=1, dt=4, t0=0): return [step(t, a0, a1, dt, t0) for t in time]


"""
Naive numerical differentiation
"""
def num_jacobian(X, U, P, dyn):
    s_size = len(X)
    i_size = len(U)
    epsilonX = (0.1*np.ones(s_size)).tolist()
    dX = np.diag(epsilonX)
    A = np.zeros((s_size, s_size))
    for i in range(0, s_size):
        dx = dX[i,:]
        delta_f = dyn(X+dx/2, 0, U, P) - dyn(X-dx/2, 0, U, P)
        delta_f = delta_f / dx[i]
        A[:,i] = delta_f

    epsilonU = (0.1*np.ones(i_size)).tolist()
    dU = np.diag(epsilonU)
    B = np.zeros((s_size,i_size))
    for i in range(0, i_size):
        du = dU[i,:]
        delta_f = dyn(X, 0, U+du/2, P) - dyn(X, 0, U-du/2, P)
        delta_f = delta_f / du[i]
        B[:,i] = delta_f

    return A,B
