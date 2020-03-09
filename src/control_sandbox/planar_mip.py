#!/usr/bin/env python
#-*- coding: utf-8 -*-

'''
  A planar Inverted Mobile Pendulum
  See: https://poine.github.io/control_sandbox/planar_mip.html
'''

import pdb, sys
import numpy as np, math, scipy.integrate, matplotlib.pyplot as plt

import planar_mip_utils as pmip_u, misc_utils as mu

#
# Dynamic Model Parameters
#
class Param:
    def __init__(self, sat=None):
        self.R, self.L       = 0.04, 0.1      # radius of wheel and dist from wheel axis to body center of mass in m
        self.mw, self.mb     = 0.02, 0.2      # mass of the wheel and body in kg
        self.Iw, self.Ib     = 0.00025, 0.003 # inertia of wheel and body 
        self.tsat = sat or float('inf')       # max torque of the motor
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

# Numerical Jacobians of the Dynamic Model
def num_jacobian(X, U, P):
    return mu.num_jacobian(X, U, P, dyn)

class Plant():
    def __init__(self, P=None):
        self.P = P or Param()

    def cont_dyn(self, X, t, U):
        return dyn(X, t, U, self.P)
        
    def disc_dyn(self, Xk, tk, dt, Uk):
        Xkp1 = scipy.integrate.odeint(dyn, Xk, [tk, tk+dt], args=(Uk, self.P))[1]
        return Xkp1

    def num_jacobian(self, X, t, U):
        return mu.num_jacobian(X, U, self.P, dyn)

    def sim_with_input_fun(self, time, ctl, X0):
        X = np.zeros((len(time), s_size))
        U = np.zeros((len(time), iv_size))
        X[0] = X0
        for i in range(1, len(time)):
            U[i-1] = ctl.get(X[i-1], i-1)
            X[i] = self.disc_dyn(X[i-1], time[i-1], time[i]-time[i-1], [U[i-1]])
        U[-1] = U[-2]
        return X, U
    

def sim_open_loop(X0=[0, 0, 0, 0]):
    P, U = Param(), [0]
    time = np.arange(0., 7.9, 0.01)
    X = scipy.integrate.odeint(dyn, X0, time, args=(U, P ))
    U = np.zeros((len(time), 1))
    return time, X, U, P


def main(save_anim=True):
    time, X, U, P = sim_open_loop(X0=[0, 0.01, 0, 0])
    exp_name = 'open_loop'
    #pmip_u.plot(time, X, U, Yc=None, P=None)
    #pmip_u.plot2(time, X, U, Yc=None, P=None)
    #anim = pmip_u.animate(time, X, U, None, P, 'Open Loop')
    #anim = pmip_u.animate_and_plot(time, X, U, None, P, exp_name, _drawings=True, _imgs=True)
    anim = pmip_u.animate_and_plot2(time, X, U, None, P, exp_name, _drawings=True, _imgs=True)
    if save_anim:
        pmip_u.save_animation(anim, 'mip_{}.mp4'.format(exp_name), time[1]-time[0])
        # ffmpeg -i src/mip_open_loop.mp4 -r 15 docs/plots/planar_mip_sim_open_loop.gif
        #pmip_u.save_animation(anim, 'mip_{}.gif'.format(exp_name), time[1]-time[0])
    plt.show()


if __name__ == "__main__":
    main(save_anim='-save'in sys.argv)
