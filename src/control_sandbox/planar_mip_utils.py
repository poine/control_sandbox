import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image, matplotlib.offsetbox, matplotlib.transforms

import pat3.plot_utils as p3_pu

import planar_mip as pmip, misc_utils as mu
import pdb


#
# TODO: zorder not working
#

#
# a plot function
#
def plot(time, X, U, Yc=None, P=None, window_title="trajectory", figure=None, axs=None, label=None):
    #print('in plot {} {}'.format(figure, axs))
    margins=(0.04, 0.05, 0.98, 0.96, 0.20, 0.34)
    figure = figure or p3_pu.prepare_fig(figure, window_title, figsize=(0.75*20.48, 0.75*10.24), margins=margins)
    plots = [("x", "m", X[:,pmip.s_x]),
             ("$\\theta$", "deg", np.rad2deg(X[:,pmip.s_theta])),
             ("$\\tau$", "N.m", U[:,pmip.iv_t])]
    axs = axs if axs is not None else figure.subplots(3,1)
    #pdb.set_trace()
    for ax, (title, ylab, data) in zip(axs, plots):
        ax.plot(time, data, linewidth=2, label=label)
        p3_pu.decorate(ax, title=title, ylab=ylab)
    if Yc is not None:
        axs[0].plot(time, Yc, 'g', label='Setpoint')
    if P is not None and P.tsat != float('inf'):
        axs[2].plot(time, P.tsat*np.ones_like(time), 'k', label='sat')
        axs[2].plot(time, -P.tsat*np.ones_like(time), 'k', label='-sat')
    return figure, axs

def plot2(time, X, U, Yc=None, P=None, window_title="trajectory", figure=None, axs=None):
    margins=(0.04, 0.05, 0.98, 0.96, 0.20, 0.34)
    figure = figure or p3_pu.prepare_fig(figure, window_title, figsize=(0.75*20.48, 0.75*10.24), margins=margins)
    plots = [("$x$", "m", X[:,pmip.s_x]),
             ("$\\theta$", "deg", np.rad2deg(X[:,pmip.s_theta])),
             ("$\dot{x}$", "m/s", X[:,pmip.s_xd]),
             ("$\dot{\\theta}$", "deg/s", np.rad2deg(X[:,pmip.s_thetad])),
             ("$\\tau$", "N.m", U[:,pmip.iv_t])]
    axs = axs if axs is not None else figure.subplots(3,2).flatten()
    for ax, (title, ylab, data) in zip(axs, plots):
        ax.plot(time, data, linewidth=2)
        p3_pu.decorate(ax, title=title, ylab=ylab)
    if Yc is not None:
        axs[0].plot(time, Yc, 'g', label='Setpoint')
    return figure
    
#
# Display animation
#
def animate(time, X, U, Yc, P, title=None, _drawings=False, _imgs=True, figure=None, ax=None):
    dt = time[1]-time[0]
    # scene dimensions in world units
    _xmin, _xmax =  1.05*np.min(X[:,pmip.s_x])-P.R-0.5*P.L, 1.05*np.max(X[:,pmip.s_x])+P.R+0.5*P.L
    _ymin, _ymax = -1.01*(P.R+P.L), 1.6*(P.R+P.L)
    _ymin = -0.01
    fig = figure or plt.figure(figsize=(20.48, 5.12))
    if ax is None:
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(_xmin, _xmax),
                             ylim=(_ymin, _ymax), facecolor=(0.5, 0.9, 0.9))
    else:
        ax.set_xlim(_xmin, _xmax)
        ax.set_ylim(_ymin, _ymax)
        ax.set_facecolor((0.5, 0.9, 0.9))
        ax.set_aspect('equal')
        
        
    if title is not None: ax.set_title(title, {'fontsize': 20 })
    ax.grid()
    
    time_template = 'time = {:0.1f}s'
    time_text = ax.text(0.025, 0.92, '', transform=ax.transAxes)

    if _imgs:
        im_wheel = matplotlib.image.imread(mu.cs_asset('data/mip_toy_wheel.png'))
        _im_wheel = ax.imshow(im_wheel, interpolation='none',
                              origin='bottom',
                              extent=[-P.R, P.R, 0, 2*P.R], clip_on=True, alpha=0.9, zorder=3)
        im_body = matplotlib.image.imread(mu.cs_asset('data/mip_toy_body.png'))
        _bw = 2.45*P.L
        _im_body = ax.imshow(im_body, interpolation='none',
                             origin='bottom',
                             extent=[-_bw/2, _bw/2, 0, _bw], clip_on=True, alpha=0.8, zorder=3)

    _line_body, = ax.plot([], [], 'o-', lw=3, color='r', zorder=1)
    if _drawings:
        _circle_wheel = plt.Circle((0, P.R), P.R, color='r', fill=False, zorder=1)
        _line_wheel = ax.add_artist(_circle_wheel)
        if Yc is not None:
            _circle_goal = plt.Circle((0, P.R), 0.15*P.R, color='g', fill=True, zorder=1)
            _line_goal = ax.add_artist(_circle_goal)

    def init():
        #print('in init')
        #if _drawings:
            # _line_wheel ??
        time_text.set_text('')
        _line_body.set_data([], [])
        res = time_text, _line_body
        if _imgs:
            _im_wheel.set_transform(matplotlib.transforms.Affine2D())
            _im_body.set_transform(matplotlib.transforms.Affine2D())
            res += (_im_wheel, _im_body)
        if _drawings:
            res += (_line_wheel,)
            if Yc is not None: res += (_line_goal, )
        return res
        

    
    def animate(i):
        x, theta, phi = X[i, pmip.s_x], X[i, pmip.s_theta], -X[i, pmip.s_x]/P.R
        _c = np.array([x, P.R])
        _p = _c + [P.L * np.sin(-theta), P.L * np.cos(-theta)]
        if _drawings:
            _circle_wheel.center = _c
            if Yc is not None: _circle_goal.center = [Yc[i], P.R]
        _line_body.set_data([_c[0], _p[0]], [_c[1], _p[1]])

        time_text.set_text(time_template.format(i * dt))
        res = time_text, _line_body
        if _imgs:
            trans_data = matplotlib.transforms.Affine2D().rotate_around(0, P.R, phi).translate(X[i, pmip.s_x], 0) + ax.transData
            _im_wheel.set_transform(trans_data)
            dx, dz = -0.006, -3.83*P.R
            trans_data2 = matplotlib.transforms.Affine2D().translate(dx, dz).rotate_around(0, P.R, theta+np.pi).translate(X[i, pmip.s_x], 0) + ax.transData
            _im_body.set_transform(trans_data2)
            res += (_im_wheel, _im_body)
        if _drawings:
            res += (_line_wheel, )
            if Yc is not None: res = (_line_goal, )+res
        return res
    dt_mili = dt*1000#25
    anim = animation.FuncAnimation(fig, animate, np.arange(1, len(time)),
                                   interval=dt_mili, blit=True, init_func=init, repeat_delay=200)
    return anim

#
# Display animation
#
def animate_and_plot(time, X, U, Yc, P, title=None, _drawings=False, _imgs=True):
    figure, _axiss = plt.subplots(4, 1, sharex=False, sharey=False, figsize=(10.24, 10.24))
    figure = plot(time, X, U, Yc, P, window_title=title, figure=figure, axs=_axiss[1:])
    plt.tight_layout(pad=1.08, h_pad=1.08, w_pad=1.08, rect=(0, 0, 1, 0.98))
    return animate(time, X, U, Yc, P, title, _drawings, _imgs, figure=figure, ax=_axiss[0])

def animate_and_plot2(time, X, U, Yc, P, title=None, _drawings=False, _imgs=True):
    figure, _axiss = plt.subplots(4, 2, sharex=False, sharey=False, figsize=(10.24, 10.24))
    figure = plot2(time, X, U, Yc, P, window_title=title, figure=figure, axs=_axiss[1:,:].flatten())
    _axiss[3,1].remove()
    plt.tight_layout(pad=1.08, h_pad=1.08, w_pad=1.08, rect=(0, 0, 1, 0.98))
    ax_anim = plt.subplot(4, 1, 1)
    return animate(time, X, U, Yc, P, title, _drawings, _imgs, figure=figure, ax=ax_anim)

def save_animation(anim, filename, dt):
    print('encoding animation video, please wait, it will take a while')
    anim.save(filename, writer='ffmpeg', fps=1./dt)
    #anim.save(filename, fps=1./dt, writer='imagemagick') # fails... 
    print('video encoded, saved to {}, Bye'.format(filename))

