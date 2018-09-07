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
import scipy.ndimage as ndi
import cv2


def pointsetTransform(points, hgmat):
    """
    Apply transform to the positions of a point set.
    """

    points_reformatted = po.cart2homo(points)
    points_transformed = po.homo2cart(cv2.transform(points_reformatted, hgmat))

    return points_transformed


def vertexGenerator(center, fixedvertex, arot, direction=-1, scale=1, rand_amp=0, ret='all'):
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
    fixedvertex += rand_amp * np.random.uniform(high=1, low=-1, size=fixedvertex.shape)
    fixedvertex_reformatted = po.cart2homo(fixedvertex)

    if ret == 'all':
        vertices = [fixedvertex]
    elif ret == 'generated':
        vertices = []

    # Augment the scale value into an array
    if type(scale) in (int, float):
        scale = np.ones((nangles,)) * scale

    # Generate reference points by rotation and scaling
    for ira, ra in enumerate(rotangles):

        rmat = cv2.getRotationMatrix2D(center, ra, scale[ira])
        rotvertex = np.squeeze(cv2.transform(fixedvertex_reformatted, rmat)).tolist()
        vertices.append(rotvertex)

    return np.asarray(vertices, dtype='float32')


def _symcentcost(pts, center, mean_center_dist, mean_edge_dist, rotsym=6, weights=(1, 1, 1)):
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


def _refset(coeffs, landmarks, center, direction=1):
    """
    Calculate the reference point set.
    """

    arots, scales = coeffs.reshape((2, coeffs.size // 2))

    # Generate reference point set
    refs = vertexGenerator(center, fixedvertex=landmarks[0,:], arot=arots,
                           direction=direction, scale=scales, ret='generated')

    # Determine the homography that bridges the landmark and reference point sets
    H, _ = cv2.findHomography(landmarks, refs)
    # Calculate the actual point set transformed by the homography
    lmkwarped = np.squeeze(cv2.transform(landmarks[None,...], H))[:,:2]

    return lmkwarped, H


def _refsetcost(coeffs, landmarks, center, mcd, med, direction=-1, weights=(1, 1, 1)):
    """
    Reference point set generator cost function.

    :Parameters:
        coeffs : 1D array
            Point set generator coefficients (angle of rotation and scaling factors).
        landmarks : list/tuple
            Pixel coordinates of the landmarks.
        center : list/tuple
            Pixel coordinates of the Gamma point.
        direction : str | -1
            Direction to generate the point set, -1 (cw) or 1 (ccw).
        kwds : keyword arguments
            See symcentcost()

    :Return:
        rs_cost : float
            Value of the reference set cost function.
    """

    landmarks_warped, _ = _refset(coeffs, landmarks, center, direction=direction)
    rs_cost = _symcentcost(landmarks_warped, center, mcd, med, weights=weights)

    return rs_cost


def refsetopt(init, pts, center, mcd, med, niter=200, direction=-1, weights=(1, 1, 1), method='Nelder-Mead', **kwds):
    """ Optimization to find the optimal reference point set.
    """

    res = opt.basinhopping(_refsetcost, init, niter=niter, minimizer_kwargs={'method':method,\
                       'args':(pts, center, mcd, med, direction, weights)}, **kwds)
    # Calculate the optimal warped point set and the corresponding homography
    ptsw, H = _refset(res['x'], pts, center, direction)

    return ptsw, H


def imgWarping(img, hgmat=None, landmarks=None, refs=None, rotangle=None, **kwds):
    """
    Perform image warping based on a generic affine transform (homography).

    :Parameters:
        img : 2D array
            Input image (distorted).
        hgmat : 2D array
            Homography matrix.
        landmarks : list/array
            Pixel coordinates of landmarks (distorted).
        refs : list/array
            Pixel coordinates of reference points (undistorted).
        rotangle : float
            Rotation angle (in degrees).
        **kwds : keyword argument

    :Returns:
        imgaw : 2D array
            Image after affine warping.
        hgmat : 2D array
            (Composite) Homography matrix for the tranform.
    """

    # Calculate the homography matrix, if not given
    if hgmat is None:

        landmarks = np.asarray(landmarks, dtype='float32')
        refs = np.asarray(refs, dtype='float32')
        hgmat, _ = cv2.findHomography(landmarks, refs)

    # Add rotation to the transformation, if specified
    if rotangle is not None:

        center = kwds.pop('center', ndi.measurements.center_of_mass(img))
        center = tuple(center)
        rotmat = cv2.getRotationMatrix2D(center, angle=rotangle, scale=1)
        # Construct rotation matrix in homogeneous coordinate
        rotmat = np.concatenate((rotmat, np.array([0, 0, 1], ndmin=2)), axis=0)
        # Construct composite operation
        hgmat = np.dot(rotmat, hgmat)

    # Perform composite image transformation
    imgaw = cv2.warpPerspective(img, hgmat, img.shape)

    return imgaw, hgmat


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
