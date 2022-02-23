#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 19:05:05 2022

@author: Alexander Southan
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pyRandomWalk import random_walk

# Define the box for the random walks
limits = {'x': [-1, 1], 'y': [-1, 1]}

# Generate random walks
random_walks = random_walk(
    step_number=20, number_of_walks=2, limits=limits, wall_mode='reflect',
    step_length=0.5)


# Query the different coordinates that are used for the plots
all_coords = random_walks.get_coords('all')
walk_coords = random_walks.get_coords('walk_points')
reflect_coords = random_walks.get_coords('reflect_points')
end_coords = random_walks.get_coords('end_points')


# Plot the random walks
fig1, ax1 = plt.subplots()
for curr_walk in range(random_walks.number_of_walks):
    ax1.plot(end_coords[curr_walk][:, 0], end_coords[curr_walk][:, 1],
             ls='none', marker='o', ms='10', c='g')
    ax1.plot(walk_coords[curr_walk][:, 0], walk_coords[curr_walk][:, 1],
             ls='none', marker='o', c=['k', 'grey'][curr_walk % 2])
    ax1.plot(reflect_coords[curr_walk][:, 0], reflect_coords[curr_walk][:, 1],
             ls='none', marker='o', c='r')
    ax1.plot(all_coords[curr_walk][:, 0], all_coords[curr_walk][:, 1],
             c=['k', 'grey'][curr_walk % 2])

ax1.set_xlim([-1.1, 1.1])
ax1.set_ylim([-1.1, 1.1])
box = patches.Rectangle((random_walks.limits[0, 0], random_walks.limits[1, 0]),
                        random_walks.limits[0, 1]-random_walks.limits[0, 0],
                        random_walks.limits[1, 1]-random_walks.limits[1, 0],
                        linewidth=1, edgecolor='k', facecolor='none', ls='--')
ax1.add_patch(box)
ax1.set_aspect('equal', adjustable='box')
fig1.set_facecolor('grey')

fig1.savefig('plot walks with rectangular wall reflections.png', dpi=600)
