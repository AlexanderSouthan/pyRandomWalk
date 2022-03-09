#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 21:00:39 2022

@author: almami
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from pyRandomWalk import random_walk


random_x = np.random.random(50)*400
random_y = np.random.random(50)*100
radius = 20
start_points = np.vstack([random_x, random_y]).T

random_walks = []
for curr_start in start_points:
    random_walks.append(
        random_walk(
            step_number=10, step_length=5, number_of_walks=20,
            start_points=curr_start, dimensions=2, box_shape='circle',
            limits={'x_c': curr_start[0], 'y_c': curr_start[1], 'r': radius}))

fig1, ax1 = plt.subplots(dpi=600, tight_layout=True)
for curr_walk, curr_start in zip(random_walks, start_points):
    plt.plot(curr_walk.coords[:, :, 0].T, curr_walk.coords[:, :, 1].T,
             linewidth=0.8)
    curr_box = Circle((curr_start[0], curr_start[1]), radius=radius,
                      linewidth=0.8, edgecolor='white', facecolor='k', ls='-')
    ax1.add_patch(curr_box)

ax1.set_aspect('equal', adjustable='box')
ax1.axis('off')
ax1.margins(x=0)
fig1.set_facecolor('gray')

fig1.savefig('random_positions.png', dpi=600)
