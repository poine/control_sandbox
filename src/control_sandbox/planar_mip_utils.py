import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image, matplotlib.offsetbox, matplotlib.transforms

import pat3.plot_utils as p3_pu

import planar_mip as pmip
#
# a plot function
#
def plot(time, X, U, Y=None, figure=None, window_title="trajectory"):
    margins=(0.04, 0.05, 0.98, 0.96, 0.20, 0.34)
    figure = p3_pu.prepare_fig(figure, window_title, figsize=(0.75*20.48, 0.75*10.24), margins=margins)
    plots = [("x", "m", X[:,pmip.s_x]),
             ("$\\theta$", "deg", np.rad2deg(X[:,pmip.s_theta])),
             ("$\\tau$", "N.m", U[:,pmip.iv_t])]
    for i, (title, ylab, data) in enumerate(plots):
        ax = plt.subplot(3, 1, i+1)
        plt.plot(time, data, linewidth=2)
        p3_pu.decorate(ax, title=title, ylab=ylab)
    if Y != None:
        plt.subplot(3,1,1)
        plt.plot(time, Y, 'r')

    return figure

#
# Display animation
#
def animate(time, X, U, P, title=None, _drawings=False, _imgs=True):
    dt = time[1]-time[0]
    # scene dimensions in world units
    _xmin, _xmax = 1.05*np.min(X[:,pmip.s_x])-P.R-0.*P.L, 1.05*np.max(X[:,pmip.s_x])+P.R+0.*P.L
    _ymin, _ymax  = -1.01*(P.R+P.L), 1.6*(P.R+P.L)
    _ymin = -0.01
    fig = plt.figure(figsize=(20.48, 5.12))
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(_xmin, _xmax),
                         ylim=(_ymin, _ymax), facecolor=(0.5, 0.9, 0.9))
    
    if title is not None: ax.set_title(title, {'fontsize': 20 })
    ax.grid()
    
    time_template = 'time = {:0.1f}s'
    time_text = ax.text(0.025, 0.92, '', transform=ax.transAxes)

    if _imgs:
        im_wheel = matplotlib.image.imread('../data/mip_toy_wheel.png')
        _im_wheel = ax.imshow(im_wheel, interpolation='none',
                              origin='bottom',
                              extent=[-P.R, P.R, 0, 2*P.R], clip_on=True, alpha=0.9, zorder=1)
        im_body = matplotlib.image.imread('../data/mip_toy_body.png')
        _bw = 2.4*P.L
        _im_body = ax.imshow(im_body, interpolation='none',
                             origin='bottom',
                             extent=[-_bw/2, _bw/2, 0, _bw], clip_on=True, alpha=0.8, zorder=1)

    _line_body, = ax.plot([], [], 'o-', lw=3, color='r', zorder=2)
    if _drawings:
        _circle_wheel = plt.Circle((0, P.R), P.R, color='r', fill=False, zorder=2)
        _line_wheel = ax.add_artist(_circle_wheel)

    def init():
        print('in init')
        #if _drawings:
            # _line_wheel ??
        _line_body.set_data([], [])
            
        time_text.set_text('')
        if _imgs:
            _im_wheel.set_transform(matplotlib.transforms.Affine2D())
            _im_body.set_transform(matplotlib.transforms.Affine2D())
        res = time_text, _line_body
        if _drawings: res += (_line_wheel, )
        if _imgs: res += (_im_wheel, _im_body)
        return res
        

    
    def animate(i):
        x, theta = X[i, pmip.s_x], X[i, pmip.s_theta]
        _c = np.array([x, P.R])
        _p = _c + [P.L * np.sin(-theta), P.L * np.cos(-theta)]
        if _drawings:
            _circle_wheel.center = _c
        _line_body.set_data([_c[0], _p[0]], [_c[1], _p[1]])

        time_text.set_text(time_template.format(i * dt))
        phi = -X[i, pmip.s_x]/P.R

        if _imgs:
            trans_data = matplotlib.transforms.Affine2D().rotate_around(0, P.R, phi).translate(X[i, pmip.s_x], 0) + ax.transData
            _im_wheel.set_transform(trans_data)
            trans_data2 = matplotlib.transforms.Affine2D().translate(-0.005, -3.7*P.R).rotate_around(0, P.R, theta+np.pi).translate(X[i, pmip.s_x], 0) + ax.transData
            _im_body.set_transform(trans_data2)
        
        res = time_text, _line_body
        if _drawings: res += (_line_wheel, )
        if _imgs: res += (_im_wheel, _im_body)
        return res
    dt_mili = dt*1000#25
    anim = animation.FuncAnimation(fig, animate, np.arange(1, len(time)),
                                   interval=dt_mili, blit=True, init_func=init)
    return anim

#
# Display animation
#
def animate_and_plot(time, X, U, P, title=None, _drawings=False, _imgs=True):
    plot(time, X, U, window_title=title)
    return animate(time, X, U, P, title, _drawings, _imgs)

def save_animation(anim, filename, dt):
    print('encoding animation video, please wait, it will take a while')
    anim.save(filename, writer='ffmpeg', fps=1./dt)
    print('video encoded, Bye')

