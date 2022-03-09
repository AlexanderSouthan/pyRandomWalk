#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 19:05:05 2022

@author: Alexander Southan
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pyRandomWalk import random_walk

# Define the box for the random walks
limits = {'polygon_x': [0.4, 0.6, 0.65, 0.7, 0.9, 0.72, 0.8, 0.65, 0.6, 0.6],
          'polygon_y': [0.2, 0.3, 0.5, 0.3, 0.2, 0.15, 0, 0.15, 0, 0.15]}

# Generate random walks
random_walks = random_walk(
    step_number=200, number_of_walks=1, limits=limits, wall_mode='exclude',
    step_length=0.05, box_shape='polygon', start_points=[0.7, 0.2])

# Query the different coordinates that are used for the plots
all_coords = random_walks.get_coords('all')
walk_coords = random_walks.get_coords('walk_points')
end_coords = random_walks.get_coords('end_points')


# Plot the random walks
fig1, ax1 = plt.subplots()
for curr_walk in range(random_walks.number_of_walks):
    ax1.plot(end_coords[curr_walk][:, 0], end_coords[curr_walk][:, 1],
             ls='none', marker='o', ms='10', c='g')
    ax1.plot(walk_coords[curr_walk][:, 0], walk_coords[curr_walk][:, 1],
             ls='none', marker='o', c=['k', 'grey'][curr_walk % 2])
    ax1.plot(all_coords[curr_walk][:, 0], all_coords[curr_walk][:, 1],
              c=['k', 'grey'][curr_walk % 2])

ax1.set_xlim([0.4, 0.9])
ax1.set_ylim([0, 0.5])
box = patches.Polygon(np.array([limits['polygon_x'], limits['polygon_y']]).T,
                      linewidth=1, edgecolor='k', facecolor='none', ls='--')
ax1.add_patch(box)
ax1.set_aspect('equal', adjustable='box')
fig1.set_facecolor('grey')

fig1.savefig('plot walks with polygon wall exclusion.png', dpi=600)
