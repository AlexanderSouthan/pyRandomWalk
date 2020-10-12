# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cycler
from mpl_toolkits.mplot3d import Axes3D


class random_walk():

    def __init__(self, step_number=100, number_of_walks=1, start_x=0,
                 start_y=0, start_z=0, dimensions=2, step_length=1,
                 angles_xy=None, angles_xz=None, angles_xy_p=None,
                 angles_xz_p=None):

        # General parameters for random walk calculation
        self.step_number = step_number
        self.dimensions = dimensions
        self.number_of_walks = number_of_walks

        # Start coordinates for random walks.
        # Can be a scalar or a list.
        self.start_x = start_x
        self.start_y = start_y
        self.start_z = start_z

        # Parameters which define the change of coordinates with each step.
        # Given in polar coordinates, 0 <= angles_xy <= 2*Pi;
        # 0<= angles_xz <= Pi
        self.step_length = step_length
        self.angles_xy = angles_xy
        self.angles_xz = angles_xz

        # Probabilities of the different angles
        if angles_xy_p is not None:
            self.angles_xy_p = angles_xy_p/np.sum(angles_xy_p)
        else:
            self.angles_xy_p = angles_xy_p

        if angles_xz_p is not None:
            self.angles_xz_p = angles_xz_p/np.sum(angles_xz_p)
        else:
            self.angles_xz_p = angles_xz_p

        self.generate_walk_coordinates()

        # End to end distances are calculated as Euclidean distance (real) and
        # as square root of mean of squared differences
        self.end2end_real = np.sqrt((self.x[0, :] - self.x[-1, :])**2 +
                                    (self.y[0, :] - self.y[-1, :])**2 +
                                    (self.z[0, :] - self.z[-1, :])**2)
        self.end2end = np.sqrt(np.mean((self.x[0, :] - self.x[-1, :])**2) +
                               np.mean((self.y[0, :] - self.y[-1, :])**2) +
                               np.mean((self.z[0, :] - self.z[-1, :])**2))

    def generate_walk_coordinates(self):
        # Calculation of angles when all angles are allowed with uniform
        # probability density function
        if self.angles_xy is None:
            if self.dimensions == 1:
                self.random_walk_angles_xy = np.random.choice(
                    [0, np.pi], size=(self.step_number, self.number_of_walks))
            elif self.dimensions == 2 or self.dimensions == 3:
                self.random_walk_angles_xy = np.random.uniform(
                    0, 2*np.pi, (self.step_number, self.number_of_walks))

            if self.dimensions == 1 or self.dimensions == 2:
                self.random_walk_angles_xz = np.full(
                    ((self.step_number, self.number_of_walks)), np.pi/2)
            elif self.dimensions == 3:
                self.random_walk_angles_xz = np.random.uniform(
                    0, 2*np.pi, (self.step_number, self.number_of_walks))
        # Calculation of angles when only specific angles are allowed with
        # optional non-uniform probability density function
        else:
            if self.dimensions == 1:
                self.random_walk_angles_xy = np.random.choice(
                    [0, np.pi],
                    size=(self.step_number, self.number_of_walks),
                    p=self.angles_xy_p)
            elif self.dimensions == 2 or self.dimensions == 3:
                self.random_walk_angles_xy = np.random.choice(
                    self.angles_xy,
                    size=(self.step_number, self.number_of_walks),
                    p=self.angles_xy_p)

            if self.dimensions == 1 or self.dimensions == 2:
                self.random_walk_angles_xz = np.full(
                    ((self.step_number, self.number_of_walks)), np.pi/2)
            elif self.dimensions == 3:
                self.random_walk_angles_xz = np.random.choice(
                    self.angles_xz,
                    size=(self.step_number, self.number_of_walks),
                    p=self.angles_xz_p)

        # Polar coordinates of all individual, single steps are converted to
        # cartesian coordinates
        random_walk_individual_x = (np.cos(self.random_walk_angles_xy) *
                                    np.sin(self.random_walk_angles_xz) *
                                    self.step_length)
        random_walk_individual_x = np.insert(
            random_walk_individual_x, 0, self.start_x, axis=0)
        random_walk_individual_y = (np.sin(self.random_walk_angles_xy) *
                                    np.sin(self.random_walk_angles_xz) *
                                    self.step_length)
        random_walk_individual_y = np.insert(
            random_walk_individual_y, 0, self.start_y, axis=0)
        random_walk_individual_z = (np.cos(self.random_walk_angles_xz) *
                                    self.step_length)
        random_walk_individual_z = np.insert(
            random_walk_individual_z, 0, self.start_z, axis=0)

        # All individual steps are summed up to give the random walk
        # coordinates
        self.x = np.cumsum(random_walk_individual_x, axis=0)
        self.y = np.cumsum(random_walk_individual_y, axis=0)
        self.z = np.cumsum(random_walk_individual_z, axis=0)


n=2000
# random_walk = random_walk(
#     step_number=100,
#     step_length=1,
#     number_of_walks=n,
#     start_x=3,
#     start_y=10,
#     start_z=2,
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
    step_number=100,
    number_of_walks=n)
print(random_walk.end2end)

# color = mpl.cm.summer(np.linspace(0, 0.9, n))
# mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

fig1, ax1 = plt.subplots(1, figsize=(8,3), dpi=300)
ax1.plot(random_walk.x, random_walk.y, ls='-', lw=1)
ax1.set_axis_off()
fig1.set_facecolor('grey')
fig1.savefig('random_walk.png')

# fig2 = plt.figure()
# ax2 = fig2.gca(projection='3d')
# ax2.plot(xs=random_walk.x[:, 0],
#          ys=random_walk.y[:, 0],
#          zs=random_walk.z[:, 0])
# plt.figure(1)
# plt.hist(random_walk.end2end_real,bins=100)
