#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: R. Patrick Xian
"""

from symmetrize import pointops as po
import unittest as unit
import numpy as np


class TestPointops(unit.TestCase):
    """ Class for unit testing the symmetrize.pointops module.
    """

    def test_conversions(self):
        """ Testing the conversion between Cartesian and homogeneous coordinates.
        """

        coord_cart_single = np.array([1, 2])
        coord_homo_single = np.array([1, 2, 1])
        coord_cart_multiple = np.array([[1.5, 2.], [2.3, 10.], [11., 18.]])
        coord_homo_multiple = np.array([[1.5, 2., 1.], [2.3, 10., 1.], [11., 18., 1.]])

        # Test the single coordinate case
        self.assertTrue(np.allclose(po.cart2homo(coord_cart_single), coord_homo_single))
        self.asserttrue(np.allclose(po.homo2cart(coord_homo_single), coord_cart_single))
        # Test the multiple coordinate case
        self.assertTrue(np.allclose(po.cart2homo(coord_cart_multiple), coord_homo_multiple))
        self.assertTrue(np.allclose(po.homo2cart(coord_homo_multiple), coord_cart_multiple))

    def test_distances(self):
        """ Testing the distance calculations.
        """

        cent_1 = np.array([0.5, 0.5])
        verts_1 = np.array([[0., 1.], [0., 0.], [1., 0.], [1., 1.]])
        cent_2 = cent_1 - 0.5
        verts_2 = verts_1 - np.array([0.5., 0.5.])

        # Compare the center-vertex distances between point sets with rigidly shifted coordinates
        self.assertTrue(all(po.cvdist(verts_1, cent_1), po.cvdist(verts_2, cent_2)))
        # Compare the vertex-vertex distances between point sets with rigidly shifted coordinates
        self.assertTrue(all(po.vvdist(verts_1), po.vvdist(verts_2)))

    def test_polyarea(self):
        """ Testing the polygon area calculator.
        """

        xcoords, ycoords = [0, 1, 1, 0, 0], [0, 0, 1, 1, 0]
        xycoords = np.stack((xcoords, ycoords), axis=1)

        # Area calculation from separately provided x, y coordinates
        self.asserEqual(po.polyarea(x=xcoords, y=ycoords), 1.)
        # Area calculation from combined x, y coordinates
        self.assertEqual(po.polyarea(coords=xycoords), 1.)

    def test_arm(self):
        """ Testing the area retainment measure.
        """

        self.assertTrue(np.allclose(po.arm(1, 0.9), 0.099668))
        # Compare scores between a larger and a smaller area given the same original size
        self.assertTrue(po.arm(1, 0.9) > po.arm(1, 0.95))


if __name__ == '__main__':
    unit.main()
