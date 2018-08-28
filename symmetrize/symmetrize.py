#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: R. Patrick Xian
"""

from __future__ import print_function, division
import numpy as np
from numpy.linalg import norm, lstsq
import scipy.optimize as opt
from skimage.draw import line, circle, polygon
from skimage.feature import peak_local_max
import astropy.stats as astat
import photutils as pho
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
