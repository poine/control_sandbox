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
    def __init__(self, plant, dt, poles = [-20, -3.5+3.9j, -3.5-3.9j, -3.9]):
        A, B = plant.num_jacobian([0, 0, 0, 0], 0, [0])
        self.K = control.matlab.place(A, B, poles)
        print('K {}'.format(self.K))
        print('cl poles {}'.format(np.linalg.eig(A-np.dot(B, self.K))[0]))

    def get(self, X, i):
        dX = X - [self.x_sp[i], 0, 0, 0]
        return -np.dot(self.K, dX)
    
# Create a dataset by simulating a stabilized MIP and applying random setpoints
def make_controlled_training_set(plant, dt, force_remake=False, nsamples=int(50e3), max_nperiod=10, max_intensity=0.5):
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

def export_dataset_as_cvs(filename, time, Xks, Uks, desc, _input, _output):
    hdr = 'x_k, theta_k, xd_k, thetad_k, tau_k, x_k+1, theta_k+1, xd_k+1, thetad_k+1'
    with open(filename, "wb") as f:
        np.savetxt(f, np.hstack((_input, _output)), delimiter=',', header=hdr)

    
def ident_lin_reg(_input, _output):
    _prm_size = planar_mip.s_size*(planar_mip.s_size+planar_mip.iv_size)
    _smp_size = len(_input)
    Y, H = np.zeros((planar_mip.s_size*_smp_size)), np.zeros((planar_mip.s_size*_smp_size, _prm_size))
    for i in range(_smp_size):
        Y[planar_mip.s_size*i:planar_mip.s_size*(i+1)] = _output[i]
        H[planar_mip.s_size*i,   0:4]  = _input[i,:4]; H[planar_mip.s_size*i,  16]=_input[i,4]
        H[planar_mip.s_size*i+1, 4:8]  = _input[i,:4]; H[planar_mip.s_size*i+1,17]=_input[i,4]
        H[planar_mip.s_size*i+2, 8:12] = _input[i,:4]; H[planar_mip.s_size*i+2,18]=_input[i,4]
        H[planar_mip.s_size*i+3,12:16] = _input[i,:4]; H[planar_mip.s_size*i+3,19]=_input[i,4]
    params = np.dot(np.linalg.pinv(H), Y)
    return params

def validate(plant, params, dt):
    # compare jacobians
    Ac, Bc = plant.num_jacobian([0, 0, 0, 0], 0, [0])
    cont_sys = control.ss(Ac, Bc, [[1, 0, 0, 0]], [[0]])
    disc_sys = control.sample_system(cont_sys, dt)
    print('real jacobian\n{}\n{}'.format(disc_sys.A, disc_sys.B))
    print(params)
    A1d = params[:16].reshape((4,4))
    B1d = params[16:].reshape((4,1))
    print('identified jacobian\n{}\n{}'.format(A1d, B1d))
    # sim real plant
    ctl = CtlPlaceFullPoles(plant, dt)
    time = np.arange(0, 10., dt); ctl.x_sp = mu.step_vec(time, a0=-0.2, a1=0.2)
    X0 = [0., 0.01, 0, 0]
    X, U = plant.sim_with_input_fun(time, ctl, X0)
    # sim identified linear plant
    Xm, Um = np.zeros((len(time), 4)), np.zeros((len(time), 1))
    Xm[0] = X0
    for k in range(1, len(time)):
        Um[k-1] = ctl.get(Xm[k-1], k-1)
        Xm[k] = np.dot(A1d, Xm[k-1]) + np.dot(B1d, Um[k-1] ) #ann.predict(np.array([[Xm[k-1,0], Xm[k-1,1], Xm[k-1,2], Xm[k-1,3], Um[k-1,0]]]))
    # plot both trajectories
    figure, axs = planar_mip_utils.plot(time, X, U, label='real')
    planar_mip_utils.plot(time, Xm, Um, figure=figure, axs=axs, label='ann')
    plt.legend()
    
def main(dt=0.01, nb_samples=int(50e3), exp_name='foo', train_type='unif', force_remake=False):
    plant = planar_mip.Plant()
    if train_type=='unif':
        time, Xks, Uks, desc, _input, _output = make_uniform_training_set(plant, dt, force_remake=force_remake, nsamples=nb_samples)
        export_dataset_as_cvs('/tmp/planar_mip__controlled_training_traj.csv', time, Xks, Uks, desc, _input, _output)
    else: # train_type=='ctld'
        time, Xks, Uks, desc, _input, _output = make_controlled_training_set(plant, dt, force_remake=force_remake, nsamples=nb_samples)
        export_dataset_as_cvs('/tmp/planar_mip__uniform_training_traj.csv', time, Xks, Uks, desc, _input, _output)
        planar_mip_utils.plot(time, Xks, Uks)
    plot_dataset(time, Xks, Uks, train_type)
    params = ident_lin_reg(_input, _output)
    validate(plant, params, dt)
    plt.show()


if __name__ == '__main__':
    main(nb_samples=int(50e3), train_type='ctld', force_remake=True)

