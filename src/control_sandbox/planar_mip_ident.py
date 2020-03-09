#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

System Identification for Planar Mobile Inverted Pendulum

"""
import sys, os, pickle, pdb
import numpy as np, matplotlib.pyplot as plt
import control.matlab             # python control toolbox

import planar_mip, planar_mip_utils, plot_utils as plu, misc_utils as mu

# Create a dataset with uniformly distributed Xks and Uks
def make_uniform_training_set(plant, dt, force_remake=False, nsamples=int(50e3),
                              min_x=-1., max_x=1., min_theta=np.deg2rad(-45), max_theta=np.deg2rad(45),
                              min_xd=-3., max_xd=3., min_thetad=np.deg2rad(-500), max_thetad=np.deg2rad(500),
                              min_tau=-0.5, max_tau=0.5):
    filename = '/tmp/planar_mip__uniform_training_traj.pkl'
    if force_remake or not os.path.isfile(filename):
        desc = 'planar mip uniform trajectory'
        Xks = np.vstack((np.random.uniform(min_x, max_x, nsamples),          # x
                         np.random.uniform(min_theta, max_theta, nsamples),  # theta
                         np.random.uniform(min_xd, max_xd, nsamples),        # xdot
                         np.random.uniform(min_thetad, max_thetad, nsamples) # theta dot
        )).T
        Uks = np.random.uniform(min_tau, max_tau, (nsamples,1))
        Xkp1s = np.zeros((nsamples, 4))
        for k in range(nsamples):
            Xkp1s[k] = plant.disc_dyn(Xks[k], 0, dt, Uks[k])
        with open(filename, "wb") as f:
            pickle.dump([Xks, Uks, Xkp1s, desc], f)
    else:
        with open(filename, "rb") as f:
          Xks, Uks, Xkp1s, desc = pickle.load(f)
    _input =  np.hstack([Xks, Uks])
    _output = Xkp1s
    time = dt*np.arange(len(_input))
    return time, Xks, Uks, desc, _input, _output


# Pole placement linear controller
class CtlPlaceFullPoles:
    def __init__(self, plant, dt, poles = [-33, -3.5+3.9j, -3.5-3.9j, -3.9]):
        A, B = plant.num_jacobian([0, 0, 0, 0], 0, [0])
        self.K = control.matlab.place(A, B, poles)
        print('K {}'.format(self.K))
        print('cl poles {}'.format(np.linalg.eig(A-np.dot(B, self.K))[0]))

    def get(self, X, i):
        dX = X - [self.x_sp[i], 0, 0, 0]
        return -np.dot(self.K, dX)
    
# Create a dataset by simulating a stabilized MIP and applying random setpoints
def make_controlled_training_set(plant, dt, force_remake=False, nsamples=int(10*1e3), max_nperiod=10, max_intensity=0.5):
    filename = '/tmp/planar_mip__controlled_training_traj.pkl'
    if force_remake or not os.path.isfile(filename):
        desc = 'planar mip controlled trajectory'
        ctl = CtlPlaceFullPoles(plant, dt)
        time, ctl.x_sp = mu.make_random_pulses(dt, nsamples, max_nperiod=max_nperiod, min_intensity=-max_intensity, max_intensity=max_intensity)
        X0 = [0, 0, 0, 0]
        X, U = plant.sim_with_input_fun(time, ctl, X0)
        mu.save_trajectory(time, X, U, desc, filename)
    else:
        time, X, U, desc = mu.load_trajectory(filename)
    _input = np.hstack([X[:-1], U[:-1]])
    _output = X[1:]
    return time, X, U, desc, _input, _output


def plot_dataset(time, X, U, exp_name):
    margins = (0.04, 0.07, 0.98, 0.93, 0.27, 0.2)
    figure = plu.prepare_fig(figsize=(20.48, 7.68), margins=margins)
    plots = [('$x$', 'm', X[:,0]),
             ('$\\theta$', 'deg', np.rad2deg(X[:,1])),
             ('$\dot{x}$', 'm/s', X[:,2]),
             ('$\dot{\\theta}$', 'deg/s', np.rad2deg(X[:,3])),
             ('$\\tau$', 'N', U[:,0])]
    
    for i, (_ti, _un, _d) in enumerate(plots):
        ax = plt.subplot(1,5,i+1)
        plt.hist(_d, bins=100)
        plu.decorate(ax, title=_ti, xlab=_un)
    #ut.save_if('../docs/plots/plant_id__mip_simple__{}_training_set_histogram.png'.format(exp_name))


def ident_lin_reg(_input, _output):
    _prm_size = planar_mip.s_size*(planar_mip.s_size+planar_mip.iv_size)
    _smp_size = len(_input)
    Y, H = np.zeros((planar_mip.s_size*_smp_size)), np.zeros((planar_mip.s_size*_smp_size, _prm_size))
    for i in range(_smp_size):
        Y[planar_mip.s_size*i:planar_mip.s_size*(i+1)] = _output[i]
        H[planar_mip.s_size*i,:4]      = _input[i,:4]; H[planar_mip.s_size*i,16]=_input[i,4]
        H[planar_mip.s_size*i+1,4:8]   = _input[i,:4]; H[planar_mip.s_size*i,17]=_input[i,4]
        H[planar_mip.s_size*i+1,8:12]  = _input[i,:4]; H[planar_mip.s_size*i,18]=_input[i,4]
        H[planar_mip.s_size*i+1,12:16] = _input[i,:4]; H[planar_mip.s_size*i,19]=_input[i,4]
    params = np.dot(np.linalg.pinv(H), Y)
    print(params)
    A = params[:16].reshape((4,4))
    pdb.set_trace()

def main(dt=0.01, nb_samples=int(1e3)):
    plant = planar_mip.Plant()
    if 1:
        time, Xks, Uks, desc, _input, _output = make_uniform_training_set(plant, dt, force_remake=False)
    else:
        time, Xks, Uks, desc, _input, _output = make_controlled_training_set(plant, dt, force_remake=False, nsamples=int(10*1e3))
        planar_mip_utils.plot(time, Xks, Uks)
    plot_dataset(time, Xks, Uks, 'Uniform')
    ident_lin_reg(_input, _output)
    A, B = plant.num_jacobian([0, 0, 0, 0], 0, [0])
    print(A, B)
    plt.show()


if __name__ == '__main__':
    main()

