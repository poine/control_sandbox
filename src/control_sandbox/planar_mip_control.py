#!/usr/bin/env python
#-*- coding: utf-8 -*-

'''
  Feedback on a Mobile Inverted Pendulum (MIP)

'''
import pdb
import numpy as np, math, scipy.integrate, matplotlib.pyplot as plt
import control.matlab

import planar_mip as pmip
import planar_mip_utils as pmip_u
import misc_utils as cs_mu

def sim_state_feedback(X0, K, H, P, sp):
    def cl_dyn(X, t): return pmip.dyn(X, t, [-np.dot(K, X)+np.dot(H, sp(t))], P)
    time = np.arange(0., 8., 0.01)
    X = scipy.integrate.odeint(cl_dyn, X0, time)
    Yc = np.array([sp(ti) for ti in time])
    U =  np.array([-np.dot(K, Xi)+np.dot(H, Yci) for Xi, Yci in zip(X, Yc)])
    return time, X, U, Yc

def sim_place(P, X0, poles, sp):
    A, B =pmip.num_jacobian([0, 0, 0, 0], [0], P)
    #print(A, B, np.linalg.eig(A), control.matlab.ctrb(A, B))
    K = control.matlab.place(A, B, poles)
    H = cs_mu.get_precommand(A, B, [[1, 0, 0, 0]], K)
    print('gain {}'.format(K))
    print('cl poles {}'.format(np.linalg.eig(A-np.dot(B, K))[0]))
    time, X, U, Yc = sim_state_feedback(X0, K, H, P, sp)
    return time, X, U, Yc, P, 'Place'

def sim_lqr(P, X0, Q, R, sp):
    A, B = pmip.num_jacobian([0, 0, 0, 0], [0], P)
    (K, X, E) = control.matlab.lqr(A, B, Q, R)
    H = cs_mu.get_precommand(A, B, [[1, 0, 0, 0]], K)
    print('feedback gain {}'.format(K))
    print('cl poles {}'.format(np.linalg.eig(A-np.dot(B, K))[0]))
    time, X, U, Yc = sim_state_feedback(X0, K, H, P, sp)
    return time, X, U, Yc, P, 'LQR'

def step(t, a=.1, p=4., dt=0.): return a if math.fmod(t+dt, p) > p/2 else -a

''' compute nested loops gain given full state feedback specifications '''
def two_loops_gains_from_place(P, poles):
    print('cl poles {}'.format(poles))
    A, B =pmip.num_jacobian([0, 0, 0, 0], [0], P)
    K = control.matlab.place(A, B, poles)
    print('feedback gain {}'.format(K))
    k0, k1, k2, k3 = K[0]
    kil_theta, kil_thetad = -k1, -k3
    kol_x, kol_xd = k0/kil_theta, k2/kil_theta
    print('kil_theta {}, kil_thetad {}'.format(kil_theta, kil_thetad))
    print('kol_x {}, kol_xd {}'.format(kol_x, kol_xd))
    return kil_theta, kil_thetad, kol_x, kol_xd

def sim_two_loops_ctl(P, X0=[0, 0, 0, 0], sp=lambda t: step(t, a=0.9)):
    #Kil_theta, Kil_thetad, Kol_x, Kol_xd = 6, 6, 2, 4
    poles = [-5, -3.5+2.j, -3.5-2.j, -20]
    Kil_theta, Kil_thetad, Kol_x, Kol_xd = two_loops_gains_from_place(P, poles)
    def ctl(t, X, xsp):
        e_x, e_xd = X[pmip.s_x]-xsp,  X[pmip.s_xd]
        sp_theta = Kol_x*e_x + Kol_xd*e_xd
        max_theta = np.deg2rad(30)
        sp_theta = np.clip(sp_theta, -max_theta, max_theta)
        e_theta, e_thetad = X[pmip.s_theta]-sp_theta,  X[pmip.s_thetad]
        tau = Kil_theta*e_theta + Kil_thetad*e_thetad
        return [tau]
    def cl_dyn(X, t): return pmip.dyn(X, t, ctl(t, X, sp(t)), P)
    time = np.arange(0., 8, 0.01)
    X = scipy.integrate.odeint(cl_dyn, X0, time)
    U =  np.array([ctl(ti, Xi, sp(ti)) for ti, Xi in zip(time, X)])
    Yc = np.array([sp(ti) for ti in time])
    return time, X, U, Yc, P, 'Two_Loops'


def main(save_anim=False):
    P = pmip.Param(sat=0.3)
    _a = 0.4
    X0 = [_a, 0.01, 0, 0]
    if 0:
        poles = [-5, -3.5+2.j, -3.5-2.j, -20]
        time, X, U, Yc, P, exp_name = sim_place(P, X0, poles, sp=lambda t: step(t, a=_a))
    if 0:
        Q, R = np.diag([5, 1, 0.1, 0.01]), np.diag([4])
        time, X, U, Yc, P, exp_name =  sim_lqr(P, X0, Q, R, sp=lambda t: step(t, a=_a))
    if 1:
        poles = [-5, -3.5+2.j, -3.5-2.j, -20]
        time, X, U, Yc, P, exp_name =  sim_two_loops_ctl(P, X0, sp=lambda t: step(t, a=_a))
    #pmip_u.plot(time, X, U, Yc, P, window_title=exp_name)
    #anim = pmip_u.animate(time, X, U, Yc, P, exp_name)
    anim = pmip_u.animate_and_plot2(time, X, U, Yc, P, exp_name, _drawings=True, _imgs=True)
    if save_anim:
        pmip_u.save_animation(anim, 'mip_{}.mp4'.format(exp_name), time[1]-time[0])
    plt.show()
    
if __name__ == "__main__":
    main()
