#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: R. Patrick Xian
"""

# ========================= #
# Operations on point sets  #
# ========================= #

import numpy as np
from numpy.linalg import norm
from skimage.feature import peak_local_max
import astropy.stats as astat
import photutils as pho


def peakdetect2d(img, method='daofind', **kwds):
    """
    Peak detection in 2D image.

    :Parameters:
        img : 2D array
            Image matrix.
        method : str | 'daofind'
            Detection method ('daofind' or 'maxlist').
        **kwds : keyword arguments
            Arguments passed to the specific methods chosen.

    :Return:
        pks : 2D array
            Pixel coordinates of detected peaks, in (column, row) ordering.
    """

    if method == 'daofind':

        sg = kwds.pop('sigma', 5.0)
        fwhm = kwds.pop('fwhm', 3.0)
        threshfactor = kwds.pop('threshfactor', 8)

        mean, median, std = astat.sigma_clipped_stats(img, sigma=sg)
        daofind = pho.DAOStarFinder(fwhm=fwhm, threshold=threshfactor*std)
        sources = daofind(img)
        pks = np.stack((sources['ycentroid'], sources['xcentroid']), axis=1)

    elif method == 'maxlist':

        mindist = kwds.pop('mindist', 10)
        numpeaks = kwds.pop('numpeaks', 7)

        pks = peak_local_max(img, min_distance=mindist, num_peaks=numpeaks)

    return pks


def pointset_center(pset, condition='among', method='meancp'):
    """
    Determine the center position of a point set.

    :Parameters:
        pset : 2D array
            Pixel coordinates of the point set.
        condition : str | 'among'
            Condition to extract the points
            'among' = use a point among the set
            'unrestricted' = use the centroid coordinate
        method : str | 'meancp'
            Method to determine the point set center.
    """

    # Centroid position of point set
    pmean = np.mean(pset, axis=0)

    # Compare the coordinates with the mean position
    if method == 'meancp':
        dist = norm(pset - pmean, axis=1)
        minid = np.argmin(dist)
        pscenter = pset[minid, :]
        prest = np.delete(pset, minid, axis=0)
    else:
        raise NotImplementedError

    if condition == 'among':
        return pscenter, prest
    elif condition == 'unrestricted':
        return pmean


def pointset_order(pset, center=None, direction='cw'):
    """
    Order a point set around a center in a clockwise or counterclockwise way.

    :Parameters:
        pset : 2D array
            Pixel coordinates of the point set.
        center : list/tuple/1D array | None
            Pixel coordinates of the putative shape center.
        direction : str | 'cw'
            Direction of the ordering ('cw' or 'ccw').

    :Return:
        pset_ordered : 2D array
            Sorted pixel coordinates of the point set.
    """

    dirdict = {'cw':1, 'ccw':-1}

    # Calculate the coordinates of the
    if center is None:
        pmean = np.mean(pset, axis=0)
        pshifted = pset - pmean
    else:
        pshifted = pset - center

    pangle = np.arctan2(pshifted[:, 1], pshifted[:, 0]) * 180/np.pi
    # Sorting order
    order = np.argsort(pangle)[::dirdict[direction]]
    pset_ordered = pset[order]

    return pset_ordered


def vvdist(verts, neighbor=1):
    """ Calculate the neighboring vertex-vertex distance
    """

    if neighbor == 1:
        vvd = norm(verts - np.roll(verts, shift=-1, axis=0), axis=1)

    return vvd


def cvdist(verts, center):
    """ Calculate the center-vertex distance
    """

    return norm(verts - center, axis=1)
