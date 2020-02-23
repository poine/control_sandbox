#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""


class Solver:
    def __init__(self,
                 _obj_fun=obj_final_z, _obj_grad=obj_grad_final_z,
                 _atm=None,
                 _min_bank=-np.deg2rad(30.), _max_bank=np.deg2rad(30.),
                 _min_v=-1., _max_v=1.,
                 x0=0.1, th0=0):
        self.duration  =    8
        self.num_nodes =  self.duration*100 # time discretization
        self.interval_value = self.duration / (self.num_nodes - 1)
        print('solver: interval_value: {:.3f}s ({:.1f}hz)'.format(self.interval_value, 1./self.interval_value))

        # symbols
        self._st = sym.symbols('t')
        self._sx, self._sth, self._sxd, self._sthd, self._stau = sym.symbols('x, theta, xd, thetad, tau', cls=sym.Function)
        self._state_symbols = (self._sx(self._st), self._sth(self._st), self._sxd(self._st), self._sthd(self._st))
        self._input_symbols = (self._stau)

        # Specify the known system parameters.
        self._par_map = collections.OrderedDict()

        
        # Specify the symbolic instance constraints, i.e. initial and end conditions.
        self._instance_constraints = (self._sx(0.)-x0, self._sth(0.)-th0, self._sxd(0.), self._sthd(0.))
        
        self._bounds = {self._sthe(self._st): (_min_bank, _max_bank),
                        self._sxd(self._st): (_min_v, _max_v)        }
        # add theta sth
        
        # Create an optimization problem.
        self.prob =  opty.direct_collocation.Problem(lambda _free: _obj_fun(self.num_nodes, 1., _free),
                                                     lambda _free: _obj_grad(self.num_nodes, 1., _free),
                                                     self.get_eom(),
                                                     self._state_symbols,
                                                     self.num_nodes,
                                                     self.interval_value,
                                                     known_parameter_map=self._par_map,
                                                     instance_constraints=self._instance_constraints,
                                                     bounds=self._bounds)
        
    def get_eom(self, g=9.81):
        eq1 = self._sx(self._st).diff() - self._sv(self._st) * sym.cos(self._spsi(self._st))
        eq2 = self._sy(self._st).diff() - self._sv(self._st) * sym.sin(self._spsi(self._st))
        eq3 = self._sz(self._st).diff()\
              -self.atm.get_wind_sym(self._sx(self._st), self._sy(self._st), self._sz(self._st), self._st)\
              -go_u.glider_sink_rate(self._sv(self._st), self._sphi(self._st))
        eq4 = self._spsi(self._st).diff() - g / self._sv(self._st) * sym.tan(self._sphi(self._st))
        return sym.Matrix([eq1, eq2, eq3, eq4])
