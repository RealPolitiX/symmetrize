#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: R. Patrick Xian
"""

# ========================= #
# Operations on point sets  #
# ========================= #

from __future__ import print_function, division
from . import pointops as po
import numpy as np
import scipy.optimize as opt
import cv2


def vertexGenerator(center, fixedvertex, arot, direction=-1, scale=1, ret='all'):
    """
    Generation of the vertices of symmetric polygons.

    :Parameters:
        center : (int, int)
            Pixel positions of the symmetry center (row pixel, column pixel).
        fixedvertex : (int, int)
            Pixel position of the fixed vertex (row pixel, column pixel).
        arot : float
            Spacing in angle of rotation.
        direction : int | 1
            Direction of angular rotation (1 = anticlockwise, -1 = clockwise)
        scale : float
            Radial scaling factor.
        ret : str | 'all'
            Return type. Specify 'all' returns all vertices, specify 'generated'
            returns only the generated ones (without the fixedvertex in the argument).

    :Return:
        vertices : 2D array
            Collection of generated vertices.
    """

    if type(arot) in (int, float):
        nangles = int(np.round(360 / arot)) - 1 # Number of angles needed
        rotangles = direction*np.linspace(1, nangles, nangles)*arot
    else:
        nangles = len(arot)
        rotangles = np.cumsum(arot)

    # Reformat the input array to satisfy function requirement
    fixedvertex_reformatted = np.array(fixedvertex, dtype='int32', ndmin=2)[None,...]

    if ret == 'all':
        vertices = [fixedvertex]
    elif ret == 'generated':
        vertices = []

    if type(scale) in (int, float):
        scale = np.ones((nangles,)) * scale

    # Generate reference points by rotation and scaling
    for ira, ra in enumerate(rotangles):

        rmat = cv2.getRotationMatrix2D(center, ra, scale[ira])
        rotvertex = np.squeeze(cv2.transform(fixedvertex_reformatted, rmat)).tolist()
        vertices.append(rotvertex)

    return np.asarray(vertices, dtype='int32')


def symcentcost(pts, center, mean_center_dist, mean_edge_dist, rotsym=6, weights=(1, 1, 1)):
    """
    Symmetrization-centralization loss function.

    :Parameters:
        pts : list/tuple
            List/Tuple of points.
        center : list/tuple
            Center coordinates.
        mean_center_dist : float
            Mean center-vertex distance.
        mean_edge_dist : float
            Mean nearest-neighbor vertex-vertex distance.
        rotsym : int
            Order of rotational symmetry.
        weights : list/tuple/array
            Weights for the.

    :Return:
        sc_cost : float
            The overall cost function.
    """

    # Extract the point pair
    halfsym = rotsym // 2
    pts1 = pts[range(0, halfsym), :]
    pts2 = pts[range(halfsym, rotsym), :]

    # Calculate the deviation from center
    centralcoords = (pts1 + pts2) / 2
    centerdev = centralcoords - center
    f_centeredness = np.sum(centerdev**2)

    # Calculate the distance-to-center difference between all symmetry points
    centerdist = po.cvdist(pts, center)
    f_cvdist = np.sum((centerdist - mean_center_dist)**2)

    # Calculate the edge difference between all neighboring symmetry points
    edgedist = po.vvdist(pts, 1)
    f_vvdist = np.sum((edgedist - mean_edge_dist)**2)

    # Calculate the overall cost function
    weights = np.asarray(weights)
    fsymcent = np.array([f_centeredness, f_cvdist, f_vvdist])
    sc_cost = np.dot(weights, fsymcent)

    return sc_cost


def affineWarping(img, landmarks, refs, ret='image'):
    """
    Perform image warping based on a generic affine transform (homography).

    :Parameters:
        img : 2D array
            Input image (distorted)
        landmarks : list/array
            List of pixel positions of the
        refs : list/array
            List of pixel positions of regular

    :Returns:
        imgaw : 2D array
            Image after affine warping.
        maw : 2D array
            Homography matrix for the tranform.
    """

    landmarks = np.asarray(landmarks, dtype='float32')
    refs = np.asarray(refs, dtype='float32')

    maw, _ = cv2.findHomography(landmarks, refs)
    imgaw = cv2.warpPerspective(img, maw, img.shape)

    if ret == 'image':
        return imgaw
    elif ret == 'all':
        return imgaw, maw


def applyWarping(imgstack, axis, hgmat):
    """
    Apply warping transform for a stack of images along an axis

    :Parameters:
        imgstack : 3D array
            Image stack before warping correction.
        axis : int
            Axis to iterate over to apply the transform.
        hgmat : 2D array
            Homography matrix.

    :Return:
        imstack_transformed : 3D array
            Stack of images after correction for warping.
    """

    imgstack = np.moveaxis(imgstack, axis, 0)
    imgstack_transformed = np.zeros_like(imgstack)
    nimg = imgstack.shape[0]

    for i in range(nimg):
        img = imgstack[i,...]
        imgstack_transformed[i,...] = cv2.warpPerspective(img, hgmat, img.shape)

    imgstack_transformed = np.moveaxis(imgstack_transformed, 0, axis)

    return imgstack_transformed
