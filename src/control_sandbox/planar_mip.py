#!/usr/bin/env python
#-*- coding: utf-8 -*-

'''
  A planar Inverted Mobile Pendulum
'''

import pdb
import numpy as np, math, scipy.integrate, matplotlib.pyplot as plt

import planar_mip_utils as pmip_u

#
# Dynamic Model Parameters
#
class Param:
    def __init__(self, sat=None):
        self.R, self.L       = 0.04, 0.1      # radius of wheel and dist from wheel axis to body center of mass in m
        self.mw, self.mb     = 0.02, 0.2      # mass of the wheel and body in kg
        self.Iw, self.Ib     = 0.00025, 0.003 # inertia of wheel and body 
        self.tsat = float('inf') if sat is None else sat # max torque of the motor
        self.g = 9.81
        self.compute_aux()
        
    def compute_aux(self):
        self.b = self.Ib + self.mb*self.L**2
        self.c = self.Iw + (self.mw+self.mb)*self.R**2
        self.bc = self.b*self.c
        self.h = self.mb*self.R*self.L

#
# Components of the state vector
#
s_x      = 0  # horizontal pos of the wheel axis in m
s_theta  = 1  # orientation of the body in rad, 0 up
s_xd     = 2  # horizontal velocity of the wheel axis in m/s
s_thetad = 3  # rotational velocity of the body in rad/s
s_size   = 4  # dimension of the state vector

#
# Components of the input vector
#
iv_t    = 0  # torque on the wheel axis in N.m
iv_size = 1  # dimension of the input vector

#
# Dynamic model as state space representation
#
# X, U, P : state, input, param
#
# returns Xd, time derivative of state
#
def dyn(X, t, U, P):

    Xd = np.zeros(s_size)
    th, thd = X[s_theta], X[s_thetad]
    sth, cth = math.sin(th), math.cos(th)

    # saturate input
    tau = np.clip(U[iv_t], -P.tsat, P.tsat)
    # upper rows are directly found in state vector
    Xd[s_x]  = X[s_xd]
    Xd[s_theta] = thd
    # compute the lower rows
    a = P.h*cth
    i = 1./(a**2-P.bc)
    d, e = P.mb*P.g*P.L*sth - tau, P.mb*P.R*P.L*thd**2*sth + tau
    
    Xd[s_xd]  = -P.R*i*(a*d - P.b*e)
    Xd[s_thetad] = i*(-P.c*d + a*e)

    return Xd

#
# Numerical Jacobians of the Dynamic Model
#
def num_jacobian(X, U, P):
    dX = np.diag([0.01, 0.01, 0.01, 0.01])
    A = np.zeros((s_size, s_size))
    for i in range(0, s_size):
        dx = dX[i,:]
        delta_f = dyn(X+dx/2, 0, U, P) - dyn(X-dx/2, 0, U, P)
        delta_f = delta_f / dx[i]
        A[:,i] = delta_f

    dU = np.diag([0.01])
    delta_f = dyn(X, 0, U+dU/2, P) - dyn(X, 0, U-dU/2, P)
    delta_f = delta_f / dU
    B = np.zeros((s_size,iv_size))
    B[:,0] = delta_f
    return A,B



def sim_open_loop(X0=[0, 0, 0, 0]):
    P, U = Param(), [0]
    time = np.arange(0., 7.9, 0.01)
    X = scipy.integrate.odeint(dyn, X0, time, args=(U, P ))
    U = np.zeros((len(time), 1))
    return time, X, U, P


def main(save_anim=False):
    time, X, U, P = sim_open_loop(X0=[0, 0.01, 0, 0])
    exp_name = 'open_loop'
    #pmip_u.plot(time, X, U, None)
    #anim = pmip_u.animate(time, X, U, None, P, 'Open Loop')
    anim = pmip_u.animate_and_plot(time, X, U, None, P, exp_name, _drawings=True, _imgs=True)
    if save_anim:
        pmip_u.save_animation(anim, 'mip_{}.mp4'.format(exp_name), time[1]-time[0])
    plt.show()


if __name__ == "__main__":
    main()
