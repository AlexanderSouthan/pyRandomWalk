#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 20:11:09 2022

@author: Alexander Southan
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import cycler
from mpl_toolkits.mplot3d import Axes3D

from pyRandomWalk import random_walk


n=10
# random_walk = random_walk(
#     step_number=100,
#     step_length=1,
#     number_of_walks=n,
#     start_points=[3, 10, 2],
#     dimensions=3,
#     angles_xy=[np.pi/2, np.pi, 3/2*np.pi, 2*np.pi],
#     angles_xz=[np.pi/4, np.pi*3/4],
#     angles_xz_p=[1, 1])
# random_walk = random_walk(
#     step_number=100,
#     step_length=1,
#     number_of_walks=20000,
#     start_x=3,
#     start_y=10,
#     start_z=2,
#     dimensions=3)
random_walk = random_walk(
    step_number=5, wall_mode='reflect',
    number_of_walks=n, limits={'x': [-1, 1], 'y': [-1,1], 'z': [-1, 1]}, dimensions=2, step_length=6)#,angles_xy=[np.pi/2, np.pi, 3/2*np.pi, 2*np.pi])
print(random_walk.end2end(mode='euclidean'))
print('l*sqrt(n)' , random_walk.end2end(mode='mean_of_squared'))


x_values = []
y_values = []
for curr_walk in range(n):
    x_onbox = random_walk.reflect[0][curr_walk].values
    x_inbox = random_walk.coords[curr_walk, :, 0]
    x_comb = list(zip(x_onbox, x_inbox))
    curr_x = np.array([])
    for curr_tup in x_comb:
        for curr_it in curr_tup:
            curr_x = np.append(curr_x, curr_it)
    x_values.append(curr_x)

    y_onbox = random_walk.reflect[1][curr_walk].values
    y_inbox = random_walk.coords[curr_walk, :, 1]
    y_comb = list(zip(y_onbox, y_inbox))
    curr_y = np.array([])
    for curr_tup in y_comb:
        for curr_it in curr_tup:
            curr_y = np.append(curr_y, curr_it)
    y_values.append(curr_y)

fig1, ax1 = plt.subplots(1, figsize=(8,3), dpi=300)
for curr_x, curr_y in zip(x_values, y_values):
    ax1.plot(curr_x, curr_y)
# ax1.plot(np.concatenate(x_onbox), np.concatenate(y_onbox), ls='none', marker='o')

# color = mpl.cm.summer(np.linspace(0, 0.1, n))
# mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

box = patches.Rectangle((random_walk.limits[0, 0], random_walk.limits[1, 0]),
                        random_walk.limits[0, 1]-random_walk.limits[0, 0],
                        random_walk.limits[1, 1]-random_walk.limits[1, 0],
                        linewidth=1, edgecolor='k', facecolor='none', ls='--')
ax1.add_patch(box)
ax1.set_aspect('equal', adjustable='box')
# ax1.set_axis_off()
fig1.set_facecolor('grey')
fig1.savefig('random_walk_example.png')

# # fig2 = plt.figure()
# # ax2 = fig2.gca(projection='3d')
# # ax2.plot(xs=random_walk.x[:, 0],
# #          ys=random_walk.y[:, 0],
# #          zs=random_walk.z[:, 0])
# # plt.figure(1)
# # # plt.hist(random_walk.end2end_real,bins=100)