#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 19:59:07 2021

@author: Alexander Southan
"""

import numpy as np
import unittest

from src.pyRandomWalk.random_walk import random_walk


class TestRandomWalk(unittest.TestCase):

    def test_random_walk(self):

        # generate 100000 unconstrained random walks in 3D
        random_walks = random_walk(step_number=500, number_of_walks=100000,
                                   dimensions=3, step_length=0.8)

        # Make sure that the random walks contain the right number of data
        # points
        self.assertTrue(
            np.all(random_walks.coords.shape == np.array([100000, 501, 3])))

        # Check if root of mean squared end to end distance maches with
        # theoretical expectation.
        self.assertAlmostEqual(random_walks.end2end_mean, 0.8*np.sqrt(500), 1)

    def test_constrained_walk(self):

        # generate 1000 constrained random walks in 2D
        random_walks = random_walk(step_number=1000, number_of_walks=10,
                                   dimensions=2, step_length=30,
                                   limits={'x': [-1, 10], 'y': [-1.1, 1]},
                                   wall_mode='reflect')

        self.assertTrue(np.all(random_walks.coords[:, :, 0] > -1) &
                        np.all(random_walks.coords[:, :, 0] < 10))
        self.assertTrue(np.all(random_walks.coords[:, :, 1] > -1.1) &
                        np.all(random_walks.coords[:, :, 1] < 1))
