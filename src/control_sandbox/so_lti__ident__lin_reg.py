#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os, logging, numpy as np, matplotlib.pyplot as plt

import pdb

import plot_utils as pu, misc_utils as ut, so_lti


def main(omega_plant=1., xi_plant=0.5, nb_training_samples=int(1e3), make_training_set=False, verbose=False):

    plant = so_lti.IoPlantModel(omega_plant, xi_plant)

    if 0:
        plant_in = np.random.uniform(low=-1., high=1., size=(nb_training_samples, 4))
        plant_out = np.array([plant.io_dyn(y_k, y_km1, u_k, u_km1) for y_k, y_km1, u_k, u_km1 in plant_in])
    if 0: # full state, simulated samples -- broken...
        time, X, U, desc = so_lti.make_or_load_training_set(plant, ut.CtlNone(), make_training_set)
        so_lti.plot(time, X, U) 
        _n = 2*(len(time)-1)
        Y, H = np.zeros(_n), np.zeros((_n, 6))
        for i in range(len(time)-1):
            Y[2*i:2*(i+1)] = X[i+1]
            H[2*i:2*(i+1)] = [[X[i,0], X[i,1], U[i], 0, 0, 0],
                              [0, 0, 0,X[i,0], X[i,1], U[i]]]
    if 1: # full state, uniform samples
        _n = nb_training_samples
        Y, H = np.zeros((2*_n)), np.zeros((2*_n, 6))
        Xs = np.random.uniform(low=-1., high=1., size=(_n, 2))
        Us = np.random.uniform(low=-1., high=1., size=(_n, 1))
        for i in range(_n-1):
            Y[2*i:2*(i+1)] = plant.disc_dyn(Xs[i], Us[i])
            H[2*i:2*(i+1)] = [[Xs[i,0], Xs[i,1], Us[i], 0, 0, 0],
                              [0, 0, 0,Xs[i,0], Xs[i,1], Us[i]]]
    
    Xs = np.dot(np.linalg.pinv(H), Y)
    print('real plant:\n{}\n{}'.format(plant.Ad, plant.Bd))
    Ad1 = np.array([[Xs[0], Xs[1]],[Xs[3], Xs[4]]])
    Bd1 = np.array([[Xs[2]],[Xs[5]]])
    print('identified:\n{}\n{}'.format(Ad1, Bd1))
    #pdb.set_trace()

            
    
    

if __name__ == "__main__":
     logging.basicConfig(level=logging.INFO)
     main(verbose=True)
     plt.show()
