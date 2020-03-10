#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

System Identification for Planar Mobile Inverted Pendulum

"""
import sys, os, pickle, pdb
import numpy as np, matplotlib.pyplot as plt
import control.matlab             # python control toolbox
import keras

import planar_mip, planar_mip_utils, planar_mip_ident as pmi, plot_utils as plu, misc_utils as mu


def ident_plant(_input, _output, expe_name, epochs=50, force_train=False, display_training_history=False):
    filename = "/tmp/plant_id__planar_mip.h5"
    if force_train or not os.path.isfile(filename):
        plant_i = keras.layers.Input((5,), name ="plant_i") # x1_k, x2_k, x3_k, x4_k, u_k
        if 1:
            plant_l = keras.layers.Dense(4, activation='linear', kernel_initializer='uniform', use_bias=False, name="plant")
            plant_o = plant_l(plant_i)
        else:
            plant_l1 = keras.layers.Dense(8, activation='relu', kernel_initializer='uniform', use_bias=True, name="plant1")
            plant_l2 = keras.layers.Dense(12, activation='relu', kernel_initializer='uniform', use_bias=True, name="plant2")
            plant_l3 = keras.layers.Dense(4, activation='linear', kernel_initializer='uniform', use_bias=True, name="plant3")
            plant_o = plant_l3(plant_l2(plant_l1(plant_i)))
        plant_ann = keras.models.Model(inputs=plant_i, outputs=plant_o)
        plant_ann.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

        
        
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)
        history = plant_ann.fit(_input, _output, epochs=epochs, batch_size=32,  verbose=1, shuffle=True, validation_split=0.1, callbacks=[early_stopping])
        

        
        if display_training_history:
            margins = (0.04, 0.07, 0.98, 0.93, 0.27, 0.2)
            figure = plu.prepare_fig(figsize=(20.48, 7.68), margins=margins)
            ax = plt.subplot(1,2,1); plu.decorate(ax, title='loss'); plt.plot(history.history['loss'])
            ax = plt.subplot(1,2,2); plu.decorate(ax, title='accuracy'); plt.plot(history.history['acc'])
            #plu.save_if('../docs/plots/plant_id__mip_simple__{}_training_history.png'.format(expe_name))
        plant_ann.save(filename)
    else:
        plant_ann = keras.models.load_model(filename)
    return plant_ann

def validate(plant, ann, dt, expe_name):
    Ac, Bc = plant.num_jacobian([0, 0, 0, 0], 0, [0])
    cont_sys = control.ss(Ac, Bc, [[1, 0, 0, 0]], [[0]])
    disc_sys = control.sample_system(cont_sys, dt)
    print('real jacobian\n{}\n{}'.format(disc_sys.A, disc_sys.B))
    
    w = ann.get_layer(name='plant').get_weights()[0]
    A1d, B1d = w[:4], w[4].reshape(4,1)
    print('identified jacobian\n{}\n{}'.format(A1d, B1d))
    
    ctl = pmi.CtlPlaceFullPoles(plant, dt)
    time = np.arange(0, 10., dt); ctl.x_sp = mu.step_vec(time, a0=-0.2, a1=0.2)
    X0 = [0., 0.01, 0, 0]
    X, U = plant.sim_with_input_fun(time, ctl, X0)

    Xm, Um = np.zeros((len(time), 4)), np.zeros((len(time), 1))
    Xm[0] = X0
    for k in range(1, len(time)):
        Um[k-1] = ctl.get(Xm[k-1], k-1)
        Xm[k] = ann.predict(np.array([[Xm[k-1,0], Xm[k-1,1], Xm[k-1,2], Xm[k-1,3], Um[k-1,0]]]))
    figure, axs = planar_mip_utils.plot(time, X, U, label='real')
    planar_mip_utils.plot(time, Xm, Um, figure=figure, axs=axs, label='ann')
    plt.legend()
    #plt.savefig('../docs/plots/plant_id__mip_simple__{}_fs.png'.format(expe_name))

    
def main(dt=0.01, nb_samples=int(1e3), exp_name='foo'):
    plant = planar_mip.Plant()
    if 1:
        time, Xks, Uks, desc, _input, _output = pmi.make_uniform_training_set(plant, dt, force_remake=False)
    else:
        time, Xks, Uks, desc, _input, _output = pmi.make_controlled_training_set(plant, dt, force_remake=False, nsamples=int(50e3))
        planar_mip_utils.plot(time, Xks, Uks)
    pmi.plot_dataset(time, Xks, Uks, 'Uniform')
    ann = ident_plant(_input, _output, exp_name, epochs=150, force_train=False, display_training_history=True)
    validate(plant, ann, dt, exp_name)
   
    plt.show()


if __name__ == '__main__':
    main()

