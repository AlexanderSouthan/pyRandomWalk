# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from little_helpers.geometry import (
    reflect_line_in_box, point_inside_cartesianbox, point_inside_circle,
    point_inside_polygon)

class random_walk():

    def __init__(self, step_number=100, number_of_walks=1, start_points=None,
                 dimensions=2, step_length=1, angles_xy=None, angles_xz=None,
                 angles_xy_p=None, angles_xz_p=None, limits=None,
                 constraint_counter=1000, wall_mode='exclude',
                 box_shape='rectangle'):
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
            A dictionary with keys defining the box used as a constraint. These
            are different for the different box_shape values.
            For 'rectangle':
                Keys are ['x', 'y', 'z'] and entries are lists containing two
                elements defining the minimum and maximum allowed values of
                random walk coordinates. The default is None, meaning that
                there is no constraint on any coordinate (all values are
                allowed). Also, only the upper or the lower limit may be None,
                so that along that dimension a constraint is only presenst in
                one direction.
            For 'circle':
                Keys are ['x_c', 'y_c', 'z_c', 'r'] and entries are all floats.
                'x_c', 'y_c' and 'z_c' are the coordinates of the center point
                of the circle or sphere and 'r' is the radius.
            For 'polygon':
                Keys are ['polygon_x', 'polygon_y'] and entries are lists
                containing the x and y coordinates of the polygon corners.
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
            on the walls of the box defined by the constraints. The default is
            'exclude'.
        box_shape : str, optional
            The shape of the box used to define the contraints. Allowed values
            are 'rectangle' for a recangular box, 'circle' for a circular
            box and 'polygon' (circle and polygon work only for wall_mode =
           'exclude' currently). 'polygon' works only in 2D, 'circle' in 2D and
            3D, 'rectangle' in 1D, 2D and 3D. The default is 'rectangle'.

        Returns
        -------
        None.

        """
        # General parameters for random walk calculation
        self.step_number = step_number
        if dimensions in [1, 2, 3]:
            self.dimensions = dimensions
        else:
            raise ValueError('dimensions must be in [1, 2, 3].')
        self.number_of_walks = number_of_walks
        self.constraint_counter = constraint_counter
        self.wall_mode = wall_mode
        self.box_shape = box_shape

        # Start coordinates for random walks.
        if start_points is None:
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

        if self.box_shape == 'rectangle':
            if limits is None:
                self.limits = np.array(
                    [None for _ in ['x', 'y', 'z']
                     [:self.dimensions]])
            else:
                self.limits = np.array(
                    [limits[ii][:self.dimensions] for ii in ['x', 'y', 'z']
                     [:self.dimensions]])
        elif self.box_shape == 'circle':
            if self.dimensions < 2:
                raise ValueError(
                    'box_shape \'circle\' works only in 2D or 3D.')
            self.limits = limits
        elif self.box_shape == 'polygon':
            if self.dimensions != 2:
                raise ValueError('box_shape \'polygon\' works only in 2D.')
            self.limits = limits
        else:
            raise ValueError('No valid box_shape given.')

        self.generate_walk_coordinates()

    def generate_walk_coordinates(self):
        self.coords = np.zeros(
            (self.number_of_walks, self.step_number+1, self.dimensions))

        if self.wall_mode == 'reflect':
            self.reflect = []
            for curr_dim in range(self.dimensions):
                self.reflect.append(pd.DataFrame(
                    [[[] for _ in range(self.number_of_walks)]
                     for _ in range(self.step_number+1)],
                    index=np.arange(self.step_number+1),
                    columns=np.arange(self.number_of_walks)))

        self.coords[:, 0, :] = self.start_points

        for curr_step in range(self.step_number):
            curr_steps = self._calc_next_steps(
                    self.number_of_walks)
            self.coords[:, curr_step+1] = (
                self.coords[:, curr_step, :] + curr_steps)

            constraint_violated = self._check_constraints(
                self.coords[:, curr_step+1, :])

            if self.wall_mode == 'exclude':
                counter = np.zeros((self.number_of_walks))
                while any(constraint_violated):
                    curr_steps = self._calc_next_steps(
                        np.sum(constraint_violated))

                    self.coords[constraint_violated, curr_step+1] = (
                        self.coords[constraint_violated, curr_step, :] +
                        curr_steps)

                    constraint_violated = self._check_constraints(
                        self.coords[:, curr_step+1, :])

                    counter[constraint_violated] += 1
                    assert not any(counter[constraint_violated] >=
                                   self.constraint_counter), (
                                       'Maximum number of iterations caused by'
                                       ' constraints is reached. Probably, one'
                                       ' of the random walks is stuck in one '
                                       'of the edges of the allowed space.')

            elif self.wall_mode == 'reflect':
                if any(constraint_violated):
                    p_prev = self.coords[constraint_violated, curr_step]
                    p_viol = self.coords[constraint_violated, curr_step+1]

                    reflect_coords, final_points = reflect_line_in_box(
                        p_prev, p_viol, limits=dict(zip(
                            ['x', 'y', 'z'][:self.dimensions], self.limits)))
                    self.coords[constraint_violated, curr_step+1] = (
                        final_points)

                for curr_dim in range(self.dimensions):
                    walks_viol = self.reflect[curr_dim].columns[
                        constraint_violated]
                    for ii, curr_col in enumerate(walks_viol):
                        self.reflect[curr_dim].loc[curr_step+1, curr_col] = (
                            reflect_coords[ii][curr_dim])

            else:
                raise ValueError(
                    'wall_mode must either be \'reflect\' or \'exclude\', '
                    'but is \'{}\'.'.format(self.wall_mode))

    def _calc_next_steps(self, step_number):
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
                (step_number), np.pi/2)
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

    def _check_constraints(self, curr_coords):
        if self.box_shape == 'rectangle':
            return ~point_inside_cartesianbox(
                *curr_coords.T,
                **dict(zip(
                    ['x_limits', 'y_limits', 'z_limits'][:self.dimensions],
                    self.limits)))

        elif self.box_shape == 'circle':
            return ~point_inside_circle(*curr_coords.T, **self.limits)

        elif self.box_shape == 'polygon':
            return ~point_inside_polygon(*curr_coords.T, **self.limits)

    def end2end(self, mode='euclidean'):
        # End to end distances are calculated as Euclidean distance and
        # as square root of mean of squared differences
        if mode == 'euclidean':
            return np.sqrt(
                ((self.coords[:, 0, :]-self.coords[:, -1, :])**2).sum(axis=1))
        elif mode == 'mean_of_squared':
            return np.sqrt(
                ((self.coords[:, 0, :]-self.coords[:, -1, :])**2).sum(
                    axis=1).mean())

    def get_coords(self, mode):
        """
        Query function to retrieve certain coordinates of the random walks.

        This method is especially useful for generating graphical
        representations of the random walks.

        Parameters
        ----------
        mode : string
            Defines which coordinates will be returned. Allowed values are in
            ['all', 'walk_points', 'reflect_points', 'end_points']. Meanings
            are as follows:
                - 'all':
                    All coordinates of the random walks. This includes the walk
                    coordinates themselves, but also the reflection points in
                    case of a constrained walk. For an unconstrained walk,
                    'all' is equal to 'walk_points'.
                - 'walk_points':
                    Only the walk coordinates themselves.
                - 'reflect_points':
                    The reflect points for constrained walks with reflection
                    at the borders.
                - 'end_points':
                    Only the start and end points of the walks.

        Raises
        ------
        ValueError
            If no valid mode is given.

        Returns
        -------
        return_points : ndarray or list of ndarrays
            The coordinates defined by mode. For modes 'all' and
            'reflect_points', lists are returned with as many elements as
            random_walks. Each element has the shape [n, self.dimensions] with
            n as the number of points. The value of n depends on the number of
            reflection points. For modes 'walk_points' and 'end_points', an
            ndarray is returned with the shape [self.number_of_walks,
            self.step_number+1, self dimensions] or [self.number_of_walks,
            2, self dimensions], respectively.

        """
        modes = ['all', 'walk_points', 'reflect_points', 'end_points']

        if mode == modes[0]:  # 'all'
            all_points = []
            for curr_walk in range(self.number_of_walks):
                curr_coords = []
                for curr_point in range(self.step_number):
                    if self.wall_mode == 'reflect':
                        curr_reflect = np.vstack(
                            [curr_dim.loc[curr_point+1, curr_walk]
                             for curr_dim in self.reflect])
                    else:
                        curr_reflect = [[]]*self.dimensions

                    curr_coords.append(self.coords[curr_walk, [curr_point]].T)
                    curr_coords.append(curr_reflect)
                curr_coords.append(self.coords[curr_walk, [-1]].T)
                all_points.append(np.concatenate(curr_coords, axis=1).T)

            return_points = all_points

        elif mode == modes[1]:  # 'walk_points'
            return_points = self.coords

        elif mode == modes[2]:  # 'reflect_points'
            if self.wall_mode == 'reflect':
                reflect_points = []
                for curr_walk in range(self.number_of_walks):
                    reflect_points.append(np.vstack(
                        [np.concatenate(curr_dim[curr_walk])
                         for curr_dim in self.reflect]).T)
                return_points = reflect_points
            else:
                return np.array([])

        elif mode == modes[3]:  # 'end_points'
            return_points = self.coords[:, [0, -1], :]

        else:
            raise ValueError(
                'No valid mode given. Allowed modes are {}, but {} was given.'
                ''.format(modes, mode))

        return return_points
