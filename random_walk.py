# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import cycler
from mpl_toolkits.mplot3d import Axes3D


class random_walk():

    def __init__(self, step_number=100, number_of_walks=1, start_x=0,
                 start_y=0, start_z=0, dimensions=2, step_length=1,
                 angles_xy=None, angles_xz=None, angles_xy_p=None,
                 angles_xz_p=None, x_limits=[None, None],
                 y_limits=[None, None], z_limits=[None, None],
                 constraint_counter=1000, wall_mode='exclude'):
        """
        Initialize random walk instances.

        Parameters
        ----------
        step_number : int, optional
            The number of steps in each random walk. The default is 100.
        number_of_walks : int, optional
            The total number of random walks calculated. The default is 1.
        start_x : int or list of int, optional
            The start x positions of the random walks calculated. Can either be
            a single integer value (so that all random walks have the same
            starting x coordinate) or a list of integers with a length equal
            to number_of_walks (thus explicitly giving the starting x
            coordinate of each random walk). The default is 0.
        start_y : int or list of int, optional
            The start y positions of the random walks calculated. Can either be
            a single integer value (so that all random walks have the same
            starting y coordinate) or a list of integers with a length equal
            to number_of_walks (thus explicitly giving the starting y
            coordinate of each random walk). Only relevant if dimensions >= 2.
            The default is 0.
        start_z : int or list of int, optional
            The start z positions of the random walks calculated. Can either be
            a single integer value (so that all random walks have the same
            starting z coordinate) or a list of integers with a length equal
            to number_of_walks (thus explicitly giving the starting z
            coordinate of each random walk). Only relevant if dimensions == 3.
            The default is 0.
        dimensions : int, optional
            The dimensionality of the space in which the random walk is
            calculated. Allowed values are in [1, 2, 3]. The default is 2.
        step_length : float, optional
            The length of each step in the random walks. All steps have equal
            lengths. The default is 1.
        angles_xy : list of floats, optional
            A list containing the allowed angles (given in radian) in the
            xy-plane for each step in the random walks. Values should be in the
            interval [0, 2*pi]. The default is None, meaning that all angles
            are allowed.
        angles_xz : list of floats, optional
            A list containing the allowed angles (given in radian) in the
            xz-plane for each step in the random walks. Values should be in the
            interval [0, 2*pi]. The default is None, meaning that all angles
            are allowed.
        angles_xy_p : list of floats, optional
            Only relevant if angles_xy is given. Relative probabilities of
            angles in angles_xy. Thus, non-uniform probability density
            functions of allowed angles can be realized. If a list, it must
            have the same length like angles_xy. The default is None, meaning
            that all angles occur with equal probabilities.
        angles_xz_p : list of floats, optional
            Only relevant if angles_xz is given. Relative probabilities of
            angles in angles_xz. Thus, non-uniform probability density
            functions of allowed angles can be realized. If a list, it must
            have the same length like angles_xz. The default is None, meaning
            that all angles occur with equal probabilities.
        x_limits : list of float, optional
            A list containing two elements defining the minimum and maximum
            allowed values of random walk x coordinates. The default is None,
            meaning that there is no constraint on the x coordinate (all values
            are allowed).
        y_limits : list of float, optional
            A list containing two elements defining the minimum and maximum
            allowed values of random walk y coordinates. The default is None,
            meaning that there is no constraint on the y coordinate (all values
            are allowed).
        z_limits : list of float, optional
            A list containing two elements defining the minimum and maximum
            allowed values of random walk z coordinates. The default is None,
            meaning that there is no constraint on the z coordinate (all values
            are allowed).
        constraint_counter : int, optional
            With wall_mode 'exclude', this gives the maximum number of
            iterations allowed to generate new coordinates for points violating
            the constraints defined by x_limits, y_limits and z_limits. The
            default is 1000.
        wall_mode : str, optional
            Decides how to handle points that violate the constraints defined
            by x_limits, y_limits and z_limits. Allowed values are 'exclude'
            and 'reflect'. With 'exclude', new data points are calculated
            randomly until all data point satisfy the constraints (with maximum
            iterations defined in constraint_counter). With 'reflect', the
            random walks are reflected on the walls of the box defined by the
            constraints (not functional yet). The default is 'exclude'.

        Returns
        -------
        None.

        """

        # General parameters for random walk calculation
        self.step_number = step_number
        self.dimensions = dimensions
        self.number_of_walks = number_of_walks
        self.constraint_counter = constraint_counter
        self.wall_mode = wall_mode

        # Start coordinates for random walks. Can be a scalar or a list.
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

        self.x_limits = x_limits
        self.y_limits = y_limits
        self.z_limits = z_limits

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

        self.x = np.zeros((self.step_number+1, self.number_of_walks))
        self.y = np.zeros((self.step_number+1, self.number_of_walks))
        self.z = np.zeros((self.step_number+1, self.number_of_walks))

        if self.wall_mode == 'reflect':
            self.reflect_x = pd.DataFrame(
                [[[] for _ in range(self.number_of_walks)] for _ in range(self.step_number+1)],
                index=np.arange(self.step_number+1),
                columns=np.arange(self.number_of_walks))
            self.reflect_y = pd.DataFrame(
                [[[] for _ in range(self.number_of_walks)] for _ in range(self.step_number+1)],
                index=np.arange(self.step_number+1),
                columns=np.arange(self.number_of_walks))
            self.reflect_z = pd.DataFrame(
                [[[] for _ in range(self.number_of_walks)] for _ in range(self.step_number+1)],
                index=np.arange(self.step_number+1),
                columns=np.arange(self.number_of_walks))

        self.x[0] = self.start_x
        self.y[0] = self.start_y
        self.z[0] = self.start_z

        for curr_step in range(self.step_number):
            curr_x_step, curr_y_step, curr_z_step = self.calc_next_steps(
                    self.number_of_walks)
            self.x[curr_step+1] = self.x[curr_step] + curr_x_step
            self.y[curr_step+1] = self.y[curr_step] + curr_y_step
            self.z[curr_step+1] = self.z[curr_step] + curr_z_step

            constraint_violated = self.check_constraints(
                self.x[curr_step+1], self.y[curr_step+1], self.z[curr_step+1])

            if self.wall_mode == 'exclude':
                counter = np.zeros((self.number_of_walks))
                constraint_violated = np.sum(constraint_violated, axis=0, dtype='bool')
                while any(constraint_violated):
                    curr_x_step, curr_y_step, curr_z_step = self.calc_next_steps(
                        np.sum(constraint_violated))

                    self.x[curr_step+1, constraint_violated] = (
                        self.x[curr_step, constraint_violated] + curr_x_step)
                    self.y[curr_step+1, constraint_violated] = (
                        self.y[curr_step, constraint_violated] + curr_y_step)
                    self.z[curr_step+1, constraint_violated] = (
                        self.z[curr_step, constraint_violated] + curr_z_step)

                    constraint_violated = np.sum(self.check_constraints(
                        self.x[curr_step+1], self.y[curr_step+1],
                        self.z[curr_step+1]), axis=0, dtype='bool')

                    counter[constraint_violated] += 1
                    assert not any(
                        counter[constraint_violated] >= self.constraint_counter), (
                            'Maximum number of iterations caused by constraints is'
                            ' reached. Probably, one of the random walks is stuck '
                            'in one of the edges of the allowed space.')

            elif self.wall_mode == 'reflect':

                reference = []
                reference.append(np.tile(self.x[curr_step], 2).reshape(2, -1))
                reference.append(np.tile(self.y[curr_step], 2).reshape(2, -1))
                reference.append(np.tile(self.z[curr_step], 2).reshape(2, -1))

                while any(constraint_violated[0:6].ravel()):
                    for curr_dim, curr_limits, curr_coord in zip(
                            range(3),
                            [self.x_limits, self.y_limits, self.z_limits],
                            [self.x, self.y, self.z]):
                        if not all(lim is None for lim in curr_limits):
                            for ii in range(2*curr_dim, 2*curr_dim+2):
                                reflex_coords = self.calc_intersect_coords(
                                    [reference[0][ii-2*curr_dim, constraint_violated[ii]],
                                     reference[1][ii-2*curr_dim, constraint_violated[ii]],
                                     reference[2][ii-2*curr_dim, constraint_violated[ii]]],
                                    [self.x[curr_step+1, constraint_violated[ii]],
                                     self.y[curr_step+1, constraint_violated[ii]],
                                     self.z[curr_step+1, constraint_violated[ii]]],
                                    curr_limits[ii-2*curr_dim], curr_dim)

                                point_dims = [0, 1, 2]
                                point_dims.remove(curr_dim)

                                reference[curr_dim][1-ii+2*curr_dim, constraint_violated[ii]] = curr_limits[ii-2*curr_dim]
                                reference[point_dims[0]][1-ii+2*curr_dim, constraint_violated[ii]] = reflex_coords[point_dims[0]]
                                reference[point_dims[1]][1-ii+2*curr_dim, constraint_violated[ii]] = reflex_coords[point_dims[1]]

                                for curr_list_x, curr_list_y, curr_list_z, curr_reflex_x, curr_reflex_y, curr_reflex_z in zip(
                                        self.reflect_x.iloc[curr_step+1, constraint_violated[ii]],
                                        self.reflect_y.iloc[curr_step+1, constraint_violated[ii]],
                                        self.reflect_z.iloc[curr_step+1, constraint_violated[ii]],
                                        reflex_coords[0], reflex_coords[1], reflex_coords[2]):
                                    curr_list_x.append(curr_reflex_x)
                                    curr_list_y.append(curr_reflex_y)
                                    curr_list_z.append(curr_reflex_z)

                            curr_coord[curr_step+1, constraint_violated[2*curr_dim]] -= 2*(curr_coord[curr_step+1, constraint_violated[2*curr_dim]]-curr_limits[0])
                            curr_coord[curr_step+1, constraint_violated[2*curr_dim+1]] -= 2*(curr_coord[curr_step+1, constraint_violated[2*curr_dim+1]]-curr_limits[1])

                    constraint_violated = self.check_constraints(
                        self.x[curr_step+1], self.y[curr_step+1],
                        self.z[curr_step+1])

            else:
                raise ValueError(
                    'wall_mode must either be \'reflect\' or \'exclude\', '
                    'but is \'{}\'.'.format(self.wall_mode))

    def calc_next_steps(self, step_number):
        # First, the angles in the xy-plane are calculated.
        if self.dimensions == 1:
            random_walk_angles_xy = np.random.choice(
                [0, np.pi], size=(step_number), p=self.angles_xy_p)
        elif self.dimensions == 2 or self.dimensions == 3:
            if self.angles_xy is None:
                random_walk_angles_xy = np.random.uniform(0, 2*np.pi,
                                                          (step_number))
            else:
                random_walk_angles_xy = np.random.choice(
                    self.angles_xy, size=(step_number), p=self.angles_xy_p)
        else:
            raise ValueError(
                'Dimensions must be in [1, 2, 3], but is {}.'.format(
                    self.dimensions))

        # Second, the angles in the xz-plane are calculated.
        if self.dimensions == 1 or self.dimensions == 2:
            random_walk_angles_xz = np.full(
                ((step_number)), np.pi/2)
        elif self.dimensions == 3:
            if self.angles_xz is None:
                random_walk_angles_xz = np.random.uniform(0, 2*np.pi,
                                                          (step_number))
            else:
                random_walk_angles_xz = np.random.choice(
                    self.angles_xz, size=(step_number), p=self.angles_xz_p)

        # Polar coordinates are converted to cartesian coordinates
        curr_x_step = (
            np.cos(random_walk_angles_xy) * np.sin(random_walk_angles_xz) *
            self.step_length)
        curr_y_step = (
            np.sin(random_walk_angles_xy) * np.sin(random_walk_angles_xz) *
            self.step_length)
        curr_z_step = (np.cos(random_walk_angles_xz) * self.step_length)

        return (curr_x_step, curr_y_step, curr_z_step)

    def check_constraints(self, curr_x, curr_y, curr_z):
        assert len(curr_x) == len(curr_y) == len(curr_z), 'Arrays must have equal lengths.'
        constraint_violated = np.full((6, len(curr_x)), False)

        for curr_idx, (curr_values, curr_limits) in enumerate(zip(
                [curr_x, curr_y, curr_z],
                [self.x_limits, self.y_limits, self.z_limits])):
            if not all(lim is None for lim in curr_limits):
                constraint_violated[curr_idx*2] = (curr_values < curr_limits[0])
                constraint_violated[curr_idx*2+1] = (curr_values > curr_limits[1])

        return constraint_violated

    def calc_intersect_coords(self, point_1, point_2, limit, limit_dim):
        point_dims = [0, 1, 2]
        point_dims.remove(limit_dim)

        slopes_dim_1 = (point_2[point_dims[0]] - point_1[point_dims[0]])/(point_2[limit_dim] - point_1[limit_dim])
        slopes_dim_2 = (point_2[point_dims[1]] - point_1[point_dims[1]])/(point_2[limit_dim] - point_1[limit_dim])

        # coords_at_lim = [limit, slopes_dim_1*(limit-point_1[limit_dim])+point_1[point_dims[0]], slopes_dim_2*(limit-point_1[limit_dim])+point_1[point_dims[1]]]

        coords_at_lim = [0, 0, 0]
        coords_at_lim[point_dims[0]] = slopes_dim_1*(limit-point_1[limit_dim])+point_1[point_dims[0]]
        coords_at_lim[point_dims[1]] = slopes_dim_2*(limit-point_1[limit_dim])+point_1[point_dims[1]]
        coords_at_lim[limit_dim] = [limit]*len(coords_at_lim[point_dims[0]])

        return coords_at_lim

if __name__ == "__main__":

    n=2
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
        step_number=1, wall_mode='reflect',
        number_of_walks=n, x_limits=[-1, 1], y_limits=[-1,1], dimensions=2, step_length=5)#,angles_xy=[np.pi/2, np.pi, 3/2*np.pi, 2*np.pi])
    print(random_walk.end2end)
    
    # color = mpl.cm.summer(np.linspace(0, 0.9, n))
    # mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
    
    fig1, ax1 = plt.subplots(1, figsize=(8,3), dpi=300)
    
    x_values = np.array(list(zip(random_walk.reflect_x.values, random_walk.x))).ravel()
    y_values = np.array(list(zip(random_walk.reflect_y.values, random_walk.y))).ravel()
    x_plot = []
    y_plot = []
    for curr_walk in range(n):
        curr_x_combined = x_values[curr_walk::n]
        curr_y_combined = y_values[curr_walk::n]
        x_plot.append(np.concatenate([np.array([ii]).ravel() for ii in curr_x_combined]))
        y_plot.append(np.concatenate([np.array([ii]).ravel() for ii in curr_y_combined]))
    
    # x_values = np.array(list(zip(random_walk.reflect_x, random_walk.x))).ravel()
    # y_values = np.array(list(zip(random_walk.reflect_y, random_walk.y))).ravel()
    # x_values = x_values[~np.isnan(x_values)]
    # y_values = y_values[~np.isnan(y_values)]
        ax1.plot(x_plot[curr_walk], y_plot[curr_walk], ls='-', lw=1, marker='o')
        ax1.plot(np.concatenate(random_walk.reflect_x[curr_walk].values),
                 np.concatenate(random_walk.reflect_y[curr_walk].values), ls='none', marker='o', c='r')

    box = patches.Rectangle((random_walk.x_limits[0], random_walk.y_limits[0]),
                            random_walk.x_limits[1]-random_walk.x_limits[0],
                            random_walk.y_limits[1]-random_walk.y_limits[0],
                            linewidth=1, edgecolor='k', facecolor='none', ls='--')
    ax1.add_patch(box)
    ax1.set_aspect('equal', adjustable='box')
    # ax1.set_axis_off()
    fig1.set_facecolor('grey')
    fig1.savefig('random_walk.png')
    
    # fig2 = plt.figure()
    # ax2 = fig2.gca(projection='3d')
    # ax2.plot(xs=random_walk.x[:, 0],
    #          ys=random_walk.y[:, 0],
    #          zs=random_walk.z[:, 0])
    # plt.figure(1)
    # plt.hist(random_walk.end2end_real,bins=100)
