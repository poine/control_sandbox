#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute optimal trajectory for planar mobile inverted pendulum

"""
import numpy as np, sympy as sym
import matplotlib.pyplot as plt
import collections

import opty.direct_collocation

import planar_mip as pmip, planar_mip_utils as pmip_u
import pdb


class Planner:
    def __init__(self,
                 _obj_fun=None, _obj_grad=None,
                 _min_v=-2., _max_v=2.,
                 _min_bank=-np.deg2rad(30.), _max_bank=np.deg2rad(30.),
                 _min_rvel=-np.deg2rad(300.), _max_rvel=np.deg2rad(300.),
                 _min_tau=-0.3, _max_tau=0.3,
                 x0=0.1, x1=0.1, th0=0., th1=0.):
        self.P = pmip.Param()
        # Time.
        self.duration  = 2.
        self.freq = 200.                               #  discretization
        self.num_nodes =  int(self.duration*self.freq) #
        self.interval_value = self.duration / (self.num_nodes - 1)
        print('solver: interval_value: {:.3f}s ({:.1f}hz)'.format(self.interval_value, 1./self.interval_value))

        # Symbols.
        self._st = sym.symbols('t')
        self._sx, self._sth, self._sxd, self._sthd, self._stau = sym.symbols('x, theta, xd, thetad, tau', cls=sym.Function)
        self._state_symbols = (self._sx(self._st), self._sth(self._st), self._sxd(self._st), self._sthd(self._st))
        self._input_symbols = (self._stau)

        # Free variables.
        self._slice_x, self._slice_th,  self._slice_xd, self._slice_thd, self._slice_tau = \
            [slice(_i*self.num_nodes, (_i+1)*self.num_nodes, 1) for _i in range(5)]
        # Specify the known system parameters.
        self._par_map = collections.OrderedDict()

        # Cost function
        #_obj_fun  = _obj_fun  or obj_sum_tau2
        #_obj_grad = _obj_grad or obj_grad_sum_tau2
        self.R_tau = 0.001
        self.Q_x   = 1.
        self.Q_xd  = 0.
        self.Q_thd = 0.
        
        # Specify the symbolic instance constraints, i.e. initial and end conditions.
        self.x1 = x1
        self._instance_constraints = ( self._sx(0.)-x0, self._sth(0.)-th0, self._sxd(0.), self._sthd(0.),
                                       #self._sx(self.duration)-x1, self._sth(self.duration)-th1,
                                       self._sxd(self.duration), self._sthd(self.duration) )
        if 1:
            _start_pause = 100
            self._instance_constraints += tuple([self._sx(_i*self.interval_value)-x0 for _i in range(1, _start_pause)])
        if 0:
            _end_pause = 10
            self._instance_constraints += tuple([self._sx(self.duration-_i*self.interval_value)-x1 for _i in range(_end_pause)])
            self._instance_constraints += tuple([self._sth(self.duration-_i*self.interval_value)-th1 for _i in range(_end_pause)])
            self._instance_constraints += tuple([self._sxd(self.duration-_i*self.interval_value) for _i in range(_end_pause)])
            self._instance_constraints += tuple([self._sthd(self.duration-_i*self.interval_value) for _i in range(_end_pause)])

        # Specify bounds.
        self._bounds = { self._sth(self._st): (_min_bank, _max_bank),
                         self._sthd(self._st): (_min_rvel, _max_rvel),
                         self._sxd(self._st): (_min_v, _max_v),
                         self._stau(self._st): (_min_tau, _max_tau) }

        
        # Create an optimization problem.
        self.prob =  opty.direct_collocation.Problem(lambda _free: self.obj_1(_free),
                                                     lambda _free: self.obj_grad_1(_free),
                                                     self.get_eom(),
                                                     self._state_symbols,
                                                     self.num_nodes,
                                                     self.interval_value,
                                                     known_parameter_map=self._par_map,
                                                     instance_constraints=self._instance_constraints,
                                                     bounds=self._bounds)

    #
    # Dynamics
    #
    def get_eom(self):
        P = self.P
        eq1 = self._sx(self._st).diff() - self._sxd(self._st)
        eq2 = self._sth(self._st).diff() - self._sthd(self._st)
        _cth, _sth = sym.cos(self._sth(self._st)), sym.sin(self._sth(self._st))
        _a = P.h*_cth; _i = 1./(_a**2-P.bc); _b= P.mb*P.g*P.L
        _d = _b*_sth - self._stau(self._st)
        _e = _b*self._sthd(self._st)**2*_sth + self._stau(self._st)
        eq3 = self._sxd(self._st).diff() + P.R*_i*(_a*_d - P.b*_e)
        eq4 = self._sthd(self._st).diff() - _i*(-P.c*_d + _a*_e)
        return sym.Matrix([eq1, eq2, eq3, eq4])

    #
    # Cost functions
    #
    def obj_1(self, _free):
        _mean_taus = np.sum(_free[self._slice_tau]**2)/self.num_nodes
        _mean_dist_to_goal = np.sum((_free[self._slice_x]-self.x1)**2)/self.num_nodes
        _mean_vel = np.sum(_free[self._slice_xd]**2)/self.num_nodes
        _mean_rvel = np.sum(_free[self._slice_thd]**2)/self.num_nodes
        #pdb.set_trace()
        return self.R_tau*_mean_taus + self.Q_x*_mean_dist_to_goal + self.Q_xd*_mean_vel  + self.Q_thd*_mean_rvel
        
    def obj_grad_1(self, _free):
        grad = np.zeros_like(_free)
        grad[self._slice_tau] = self.R_tau * 2.*_free[self._slice_tau]/self.num_nodes
        grad[self._slice_x]   = self.Q_x   * 2.*(_free[self._slice_x]-self.x1)/self.num_nodes 
        grad[self._slice_xd]  = self.Q_xd  * 2.*_free[self._slice_xd]/self.num_nodes 
        grad[self._slice_thd] = self.Q_thd * 2.*_free[self._slice_thd]/self.num_nodes 
        return grad

    #
    # Optimization
    #
    def configure(self, tol=1e-8, max_iter=3000):
        # https://coin-or.github.io/Ipopt/OPTIONS.html
        self.prob.addOption('tol', tol)            # default 1e-8
        self.prob.addOption('max_iter', max_iter)  # default 3000
        
    def run(self):
        # Use a random positive initial guess.
        initial_guess = np.random.randn(self.prob.num_free)
        # Find the optimal solution.
        self.solution, info = self.prob.solve(initial_guess)
        self.interpret_solution()

    def interpret_solution(self):
        self.sol_time = np.linspace(0.0, self.duration, num=self.num_nodes)
        self.sol_x   = self.solution[self._slice_x]

        self.sol_X = np.vstack((self.solution[self._slice_x], self.solution[self._slice_th], self.solution[self._slice_xd], self.solution[self._slice_thd])).T
        self.sol_U = self.solution[self._slice_tau].reshape((-1,1))
        #pdb.set_trace()
        
          
def plot_solve(prob, solution):
    prob.plot_trajectories(solution)
    #prob.plot_constraint_violations(solution)
    prob.plot_objective_value()

    
def main(force_recompute=False, filename='/tmp/planar_mip_opty.pkl', save_anim=False):
    exp_name = 'opty'
    _p = Planner(x0=-1.5, x1=1.5, th0=0., th1=0.,
                 _min_bank=-np.deg2rad(30.), _max_bank=np.deg2rad(30.),
                 _min_v=-3., _max_v=3., _min_tau=-0.4, _max_tau=0.4)
    _p.configure(tol=1e-20, max_iter=3000)
    _p.run()
    plot_solve(_p.prob, _p.solution)
    #anim = pmip_u.animate(_p.sol_time, _p.sol_X, _p.sol_U, _p.P, exp_name)
    anim = pmip_u.animate_and_plot2(_p.sol_time, _p.sol_X, _p.sol_U, None, _p.P, exp_name, _drawings=True, _imgs=True)
    if save_anim:
        pmip_u.save_animation(anim, 'mip_{}.mp4'.format(exp_name), 1./_p.freq)
    plt.show()
    
if __name__ == '__main__':
    main(force_recompute=False, save_anim=False)
