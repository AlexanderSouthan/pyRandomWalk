# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import cycler
from mpl_toolkits.mplot3d import Axes3D


class random_walk():

    def __init__(self, step_number=100, number_of_walks=1, start_points=None,
                 dimensions=2, step_length=1, angles_xy=None, angles_xz=None,
                 angles_xy_p=None, angles_xz_p=None, limits=None,
                 constraint_counter=1000, wall_mode='exclude'):
        """
        Initialize random walk instances.

        Parameters
        ----------
        step_number : int, optional
            The number of steps in each random walk. The default is 100.
        number_of_walks : int, optional
            The total number of random walks calculated. The default is 1.
        start_points : 1D or 2D list of int, optional
            The start positions of the random walks calculated. Can either be
            a single point (so that all random walks have the same
            starting coordinates) or a 2D list a second dimension length equal
            to number_of_walks (thus explicitly giving the starting
            coordinate of each random walk). The default is None, meaning that
            the walks will start at the origin.
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
        limits : dict, optional
            A dictionary with keys in ['x', 'y', 'z'] and entries of lists
            containing two elements defining the minimum and maximum allowed
            values of random walk coordinates. The default is None, meaning
            that there is no constraint on any coordinate (all values are
            allowed).
        constraint_counter : int, optional
            With wall_mode 'exclude', this gives the maximum number of
            iterations allowed to generate new coordinates for points violating
            the constraints defined by limits. The default is 1000.
        wall_mode : str, optional
            Decides how to handle points that violate the constraints defined
            by limits. Allowed values are 'exclude' and 'reflect'. With
            'exclude', new data points are calculated randomly until all data
            point satisfy the constraints (with maximum iterations defined in
            constraint_counter). With 'reflect', the random walks are reflected
            on the walls of the box defined by the constraints (not functional
            yet). The default is 'exclude'.

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

        # Start coordinates for random walks.
        if start_points == None:
            self.start_points = np.zeros((1, self.dimensions))
        else:
            self.start_points = np.asarray(start_points)

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

        if limits is None:
            self.limits = np.array([[None]*self.dimensions for _ in ['x', 'y', 'z'][:self.dimensions]])
        else:
            self.limits = np.array([limits[ii][:self.dimensions] for ii in ['x', 'y', 'z'][:self.dimensions]])

        self.generate_walk_coordinates()

        # End to end distances are calculated as Euclidean distance (real) and
        # as square root of mean of squared differences
        self.end2end_real = np.sqrt(
            ((self.coords[:, 0, :]-self.coords[:, -1, :])**2).sum(axis=1))
        self.end2end = np.sqrt(
            ((self.coords[:, 0, :]-self.coords[:, -1, :])**2).sum(axis=1).mean())
            # np.mean((self.coords[:, 0, 0] - self.coords[:, -1, 0])**2) +
            # np.mean((self.coords[:, 0, 1] - self.coords[:, -1, 1])**2) +
            # np.mean((self.coords[:, 0, 2] - self.coords[:, -1, 2])**2))

    def generate_walk_coordinates(self):
        self.coords = np.zeros(
            (self.number_of_walks, self.step_number+1, self.dimensions))

        if self.wall_mode == 'reflect':
            self.reflect = []
            for curr_dim in range(self.dimensions):
                self.reflect.append(pd.DataFrame(
                    [[[] for _ in range(self.number_of_walks)] for _ in range(self.step_number+1)],
                    index=np.arange(self.step_number+1),
                    columns=np.arange(self.number_of_walks)))

        self.coords[:, 0, :] = self.start_points

        for curr_step in range(self.step_number):
            curr_steps = self.calc_next_steps(
                    self.number_of_walks)
            self.coords[:, curr_step+1] = self.coords[:, curr_step, :] + curr_steps

            constraint_violated = self.check_constraints(
                self.coords[:, curr_step+1, :])

            if self.wall_mode == 'exclude':
                counter = np.zeros((self.number_of_walks))
                constraint_violated = np.sum(constraint_violated, axis=0, dtype='bool')
                while any(constraint_violated):
                    curr_steps = self.calc_next_steps(
                        np.sum(constraint_violated))

                    self.coords[constraint_violated, curr_step+1] = (
                        self.coords[constraint_violated, curr_step, :] + curr_steps)

                    constraint_violated = np.sum(self.check_constraints(
                        self.coords[:, curr_step+1, :]), axis=0, dtype='bool')

                    counter[constraint_violated] += 1
                    assert not any(
                        counter[constraint_violated] >= self.constraint_counter), (
                            'Maximum number of iterations caused by constraints is'
                            ' reached. Probably, one of the random walks is stuck '
                            'in one of the edges of the allowed space.')

            elif self.wall_mode == 'reflect':
                constraint_violated = np.sum(constraint_violated, axis=0, dtype='bool')
                if any(constraint_violated):
                    p_prev = self.coords[constraint_violated, curr_step]
                    p_viol = self.coords[constraint_violated, curr_step+1]

                    reflect_coords, final_points, _, _, _ = reflect(
                        p_prev, p_viol, limits=dict(zip(['x', 'y', 'z'][:self.dimensions], self.limits)))
                    self.coords[constraint_violated, curr_step+1] = final_points

                for curr_dim in range(self.dimensions):
                    walks_viol = self.reflect[curr_dim].columns[constraint_violated]
                    for ii, curr_col in enumerate(walks_viol):
                        self.reflect[curr_dim].loc[curr_step+1, curr_col] = reflect_coords[ii][curr_dim]

            else:
                raise ValueError(
                    'wall_mode must either be \'reflect\' or \'exclude\', '
                    'but is \'{}\'.'.format(self.wall_mode))

    def calc_next_steps(self, step_number):
        # This method does work, but could do with some refactoring. The angles
        # have not yet been brought into an iterable format that is flexible
        # with the dimensionality of the random walks. Therefore, currently
        # the coordinates are calculated for three dimensions and only the
        # first self.dimensions are kept, the rest is thrown away: Not very
        # efficient.

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
        curr_steps = np.concatenate([
            [np.cos(random_walk_angles_xy) * np.sin(random_walk_angles_xz) *
             self.step_length],
            [np.sin(random_walk_angles_xy) * np.sin(random_walk_angles_xz) *
             self.step_length],
            [np.cos(random_walk_angles_xz) * self.step_length]],
            axis=0)[:self.dimensions].T

        return curr_steps

    def check_constraints(self, curr_coords):
        constraint_violated = np.full(
            (2*self.dimensions, curr_coords.shape[0]), False)

        for curr_idx, (curr_values, curr_limits) in enumerate(zip(
                curr_coords.T, self.limits)):
            if not all(lim is None for lim in curr_limits):
                constraint_violated[curr_idx*2] = (
                    curr_values < curr_limits[0])
                constraint_violated[curr_idx*2+1] = (
                    curr_values > curr_limits[1])

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
    limits = np.array([limits[ii] for ii in ['x', 'y', 'z'][:dimensions]]).T
    box_diff = np.abs(limits[1, :] - limits[0, :])

    # coordinates of the reflection points and the coordinate limit that causes
    # reflection
    reflect = [[[] for _ in range(dimensions)] for _ in range(start.shape[0])]
    reflect_type = [[] for _ in range(start.shape[0])]

    # calculate the intersection of the line between the points with the
    # lines of a grid formed by repeating the box limiting the allowed
    # space. This gives the coordinates of reflection points.
    for ii in range(dimensions):
        n = np.abs(direction[:, ii]) * (1/2*direction[:, ii]+1/2) #if direction[0, ii] != 0 else 0
        grid = limits[0, ii] + n*box_diff[ii]
        for curr_point in range(start.shape[0]):
            while abs(grid[curr_point]) < abs(end[curr_point, ii]):
                lambd = (grid[curr_point]-end[curr_point, ii])/point_diff[curr_point, ii]
                for jj in range(dimensions):
                    if jj != ii:
                        reflect[curr_point][jj].append(end[curr_point, jj] + lambd*point_diff[curr_point, jj])
                    else:
                        reflect[curr_point][ii].append(grid[curr_point])
                reflect_type[curr_point].append(ii)
                n[curr_point] += direction[curr_point, ii]
                grid[curr_point] = limits[0, ii] + n[curr_point]*box_diff[ii]

    # sort the reflection coordinates
    sort_idx = [np.argsort(reflect[curr_point][0])[::direction[curr_point, 0]] for curr_point in range(start.shape[0])]
    reflect = [[np.array(reflect[curr_point][ii])[sort_idx[curr_point]] for ii in range(dimensions)] for curr_point in range(start.shape[0])]
    reflect_type = [np.array(reflect_type[curr_point])[sort_idx[curr_point]] for curr_point in range(start.shape[0])]

    # Calculate the reflection points on the box faces
    re_box = [[reflect[curr_point][ii].copy() for ii in range(dimensions)] for curr_point in range(start.shape[0])]
    for curr_point in range(start.shape[0]):
        if reflect[curr_point][0].size != 0:
            for ii, r_type in enumerate(reflect_type[curr_point][:-1]):
                re_box[curr_point][r_type][ii+1:] = -(re_box[curr_point][r_type][ii+1:] -
                                                      re_box[curr_point][r_type][ii]) + re_box[curr_point][r_type][ii]

    re_box = [np.array(curr_re_box) for curr_re_box in re_box]

    # calculate the final coordinates
    final = np.zeros_like(start)
    for curr_point in range(start.shape[0]):
        if reflect_type[curr_point].size != 0:
            for ii in range(dimensions):
                if any(reflect_type[curr_point] == ii):
                    coords = re_box[curr_point][ii][reflect_type[curr_point] == ii]
                    rest = abs(point_diff[curr_point, ii]) - (
                        (reflect_type[curr_point] == ii).sum()-1)*box_diff[ii] - abs(
                            start[curr_point, ii]-coords[0])
                    if coords[-1] == limits[0, ii]:
                        final[curr_point, ii] = limits[0, ii] + rest
                    else:
                        final[curr_point, ii] = limits[1, ii] - rest
                else:
                    final[curr_point, ii] = end[curr_point, ii]

    return (re_box, final, reflect_type, reflect, direction)

if __name__ == "__main__":

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
    print(random_walk.end2end_real)
    print('l*sqrt(n)' , random_walk.end2end)


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
    fig1.savefig('random_walk.png')
    
    # # fig2 = plt.figure()
    # # ax2 = fig2.gca(projection='3d')
    # # ax2.plot(xs=random_walk.x[:, 0],
    # #          ys=random_walk.y[:, 0],
    # #          zs=random_walk.z[:, 0])
    # # plt.figure(1)
    # # # plt.hist(random_walk.end2end_real,bins=100)