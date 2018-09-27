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


def cart2homo(points):
    """ Transform from Cartesian to homogeneous coordinates.
    """

    pts_homo = np.array(points, dtype='float32', ndmin=2)[None,...]

    return pts_homo


def homo2cart(points):
    """ Transformation from homogeneous to Cartesian coordinates.
    """

    try:
        pts_cart = np.squeeze(points)[:,:2]
    except:
        pts_cart = np.squeeze(points)[:2]

    return pts_cart


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
        daofind = pho.DAOStarFinder(fwhm=fwhm, threshold=threshfactor*std, **kwds)
        sources = daofind(img)
        pks = np.stack((sources['ycentroid'], sources['xcentroid']), axis=1)

    elif method == 'maxlist':

        mindist = kwds.pop('mindist', 10)
        numpeaks = kwds.pop('numpeaks', 7)

        pks = peak_local_max(img, min_distance=mindist, num_peaks=numpeaks, **kwds)

    return pks


def pointset_center(pset, condition='among', method='centroidnn'):
    """
    Determine the center position of a point set and separate it from the rest.

    :Parameters:
        pset : 2D array
            Pixel coordinates of the point set.
        condition : str | 'among'
            Condition to extract the points
            'among' = use a point among the set
            'unrestricted' = use the centroid coordinate
        method : str | 'centroidnn' (the nearest neighbor of centroid)
            Method to determine the point set center.
    """

    # Centroid position of point set
    pmean = np.mean(pset, axis=0)

    # Compare the coordinates with the mean position
    if method == 'centroidnn':
        dist = norm(pset - pmean, axis=1)
        minid = np.argmin(dist) # The point nearest to the centroid
        pscenter = pset[minid, :] # Center coordinate
        prest = np.delete(pset, minid, axis=0) # Vertex coordinates

    elif method == 'centroid':
        pscenter = pmean
        prest = pset

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


def reorder(points, maxid, axis=0):
    """
    Reorder a point set along an axis.
    """

    pts_rolled = np.roll(points, shift=maxid-1, axis=axis)

    return pts_rolled


def rotmat(theta, to_rad=True):
    """ Rotation matrix.
    """

    theta = np.radians(theta)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    return R


def csm(pcent, pvert, rotsym):
    """
    Computation of the continuous (a)symmetry measure, bounded within [0, 1].
    When csm = 0, the point set is completely symmetric.
    When csm = 1, the point set is completely asymmetric.

    :Parameters:
        pcent : tuple/list
            Pixel coordinates of the center position.
        pvert : numpy array
            Pixel coordinates of the vertices.
        rotsym : int
            Order of rotational symmetry.

    Return:
        s : float
            Calculated continuous (a)symmetry measure.
    """

    npts = len(pvert)
    cvd = cvdist(pvert, pcent) # Center-vertex distance

    # Select the longest vector
    maxind = np.argmax(cvd)
    maxlen = cvd[maxind]

    # Calculate the normalized vector length
    cvdnorm = cvd / maxlen

    # Reorder other vectors to start with the longest
    pts_reord = reorder(pvert, maxind, axis=0)

    # Calculate the average vector length
    mcv = cvdnorm.mean()

    # Generate the rotation angles
    rotangles = 360 * (np.linspace(1, rotsym, rotsym) - 1) / rotsym

    # Calculate the unit vector along the new x axis
    xvec = pts_reord[0, :] - pcent
    xvec /= norm(xvec)

    # Rotate vector by integer multiples of symmetry angles
    devangles = [0.]
    for p, rota in zip(pts_reord[1:,], rotangles[1:]):

        R = rotmat(rota, to_rad=True)
        rotv = np.dot(R , (p - pcent).T)
        devangles.append(np.arccos(np.sum(rotv*xvec) / norm(rotv)))

    devangles = np.array(devangles)

    # Calculate the average angle
    mang = devangles.mean()

    # Calculate the distances d(Pi, Qi)
    dpq = mcv**2 + cvdnorm**2 - 2*mcv*cvdnorm*np.cos(devangles - mang)

    # Calculate the continuous asymmetry measure s
    s = dpq.sum() / npts

    return s
