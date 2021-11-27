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
                constraint_violated = np.sum(constraint_violated, axis=0, dtype='bool')

                p_prev = np.concatenate([
                   [self.x[curr_step, constraint_violated]],
                   [self.y[curr_step, constraint_violated]],
                   [self.z[curr_step, constraint_violated]]], axis=0).T

                p_viol = np.concatenate([
                    [self.x[curr_step+1, constraint_violated]],
                    [self.y[curr_step+1, constraint_violated]],
                    [self.z[curr_step+1, constraint_violated]]], axis=0).T

                x_corr = np.empty_like(self.x[curr_step+1, constraint_violated])
                y_corr = np.empty_like(self.y[curr_step+1, constraint_violated])
                z_corr = np.empty_like(self.z[curr_step+1, constraint_violated])
                for curr_idx, (curr_prev, curr_viol) in enumerate(zip(p_prev, p_viol)):
                    if self.dimensions == 2:
                        z_limi = [curr_prev[2], curr_prev[2]]
                    _, p_re, _ = reflect(
                        [curr_prev], [curr_viol], limits={'x': self.x_limits,
                                                          'y': self.y_limits,
                                                          'z': z_limi})
                    x_corr[curr_idx] = p_re[0]
                    y_corr[curr_idx] = p_re[1]
                    z_corr[curr_idx] = p_re[2]
                self.x[curr_step+1, constraint_violated] = x_corr
                self.y[curr_step+1, constraint_violated] = y_corr
                self.z[curr_step+1, constraint_violated] = z_corr


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


def reflect(start, end, limits):
    # The datapoints defining the lines to be reflected
    start = np.asarray(start)
    end = np.asarray(end)
    if start.shape == end.shape:
        dimensions = start.shape[1]
    else:
        raise ValueError(
            'Arrays for start and end point must have the same shapes.')

    # characteristics of the datapoints defining the lines to be reflected on
    # the borders of the allowed space
    point_diff = end - start
    direction = np.sign(point_diff).astype('int')

    # characteristics of the box limiting the allowed space
    limits = np.array([limits[ii] for ii in ['x', 'y', 'z'][:dimensions]]).T#pd.DataFrame.from_dict(limits)
    box_diff = np.abs(limits[1, :] - limits[0, :])
    # print(limits)


    # coordinates of the reflection points and the coordinate limit that causes
    # reflection
    reflect = [[] for _ in range(dimensions)]
    reflect_type = []

    # calculate the intersection of the line between the points with the
    # lines of a grid formed by repeating the box limiting the allowed
    # space. This gives the coordinates of reflection points.
    for ii in range(dimensions):
        n = 1/2*direction[0, ii]+1/2 if direction[0, ii] != 0 else 0
        grid = limits[0, ii] + n*box_diff[ii]
        while abs(grid) < abs(end[0, ii]):
            lambd = (grid-end[0, ii])/point_diff[0, ii]
            for jj in range(dimensions):
                if jj != ii:
                    reflect[jj].append(end[0, jj] + lambd*point_diff[0, jj])
                else:
                    reflect[ii].append(grid)
            reflect_type.append(ii)
            n += direction[0, ii]
            grid = limits[0, ii] + n*box_diff[ii]

    # sort the reflection coordinates
    sort_idx = np.argsort(reflect[0])[::direction[0, 0]]
    reflect = [np.array(reflect[ii])[sort_idx] for ii in range(dimensions)]
    reflect_type = np.array(reflect_type)[sort_idx]

    # Calculate the reflection points on the box faces
    re_box = [reflect[ii].copy() for ii in range(dimensions)]
    if reflect[0].size != 0:
        for ii, r_type in enumerate(reflect_type[:-1]):
            re_box[r_type][ii+1:] = -(re_box[r_type][ii+1:] -
                                      re_box[r_type][ii]) + re_box[r_type][ii]

    # calculate the final coordinates
    final = np.zeros(3)
    if reflect_type.size != 0:
        for ii in range(dimensions):
            if any(reflect_type == ii):
                coords = re_box[ii][reflect_type == ii]
                rest = abs(point_diff[0, ii]) - (
                    (reflect_type == ii).sum()-1)*box_diff[ii] - abs(
                        start[0, ii]-coords[0])
                if coords[-1] == limits[0, ii]:
                    final[ii] = limits[0, ii] + rest
                else:
                    final[ii] = limits[1, ii] - rest
            else:
                final[ii] = end[0, ii]

    return (re_box, final, reflect_type)

if __name__ == "__main__":

    n=1
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
