#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: R. Patrick Xian
"""

# ========================= #
# Operations on point sets  #
# ========================= #

from __future__ import print_function, division
import numpy as np
from numpy.linalg import norm
from skimage.feature import peak_local_max
import astropy.stats as astat
import photutils as pho
import matplotlib.pyplot as plt


def cart2homo(points, dtyp='float32'):
    """
    Transform points from Cartesian to homogeneous coordinates.

    :Parameter:
        points : tuple/list/array
            Pixel coordinates of the points in Cartesian coordinates, (x, y).

    :Return:
        pts_homo : 2D array
            Pixel coordinates of the points (pts) in homogeneous coordinates, (x, y, 1).
    """

    pts = np.array(points, dtype=dtyp, ndmin=2)
    ones = np.ones((len(pts), 1))
    pts_homo = np.squeeze(np.concatenate((pts, ones), axis=1))

    return pts_homo


def homo2cart(points):
    """
    Transform points from homogeneous to Cartesian coordinates.

    :Parameter:
        points : tuple/list/array
            Pixel coordinates of the points in homogeneous coordinates, (x, y, 1).

    :Return:
        pts_cart : array
            Pixel coordinates of the points (pts) in Cartesian coordinates, (x, y).
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
        ``**kwds`` : keyword arguments
            Additional arguments passed to the specific methods chosen.\n
            ``'daofind'`` See ``astropy.stats.sigma_clipped_stats()``
                            and ``photutils.detection.DAOStarFinder()``.\n
            ``'maxlist'`` See ``skimage.feature.peak_local_max()``.

    :Return:
        pks : 2D array
            Pixel coordinates of detected peaks, in (column, row) ordering.
    """

    if method == 'daofind': # DAOFind algorithm

        sg = kwds.pop('sigma', 5.0)
        fwhm = kwds.pop('fwhm', 3.0)
        threshfactor = kwds.pop('threshfactor', 8)

        mean, median, std = astat.sigma_clipped_stats(img, sigma=sg)
        daofind = pho.DAOStarFinder(fwhm=fwhm, threshold=threshfactor*std, **kwds)
        sources = daofind(img)
        pks = np.stack((sources['ycentroid'], sources['xcentroid']), axis=1)

    elif method == 'maxlist': # MaxList algorithm

        mindist = kwds.pop('mindist', 10)
        numpeaks = kwds.pop('numpeaks', 7)

        pks = peak_local_max(img, min_distance=mindist, num_peaks=numpeaks, **kwds)

    return pks


def pointset_center(pset, method='centroidnn', ret='cnc'):
    """
    Determine the center position of a point set and separate it from the rest.

    :Parameters:
        pset : 2D array
            Pixel coordinates of the point set.
        method : str | 'centroidnn' (the nearest neighbor of centroid)
            Method to determine the point set center.\n
            ``'centroidnn'`` Use the point with the minimal distance to the centroid as the center.\n
            ``'centroid'`` Use the centroid as the center.
        ret : str | 'cnc'
            Condition to extract the center position.\n
            ``'cnc'`` Return the pixel positions of the center (c) and the non-center (nc) points.\n
            ``'all'`` Return the pixel positions of the center, the non-center points and the centroid.
    """

    # Centroid position of point set
    pmean = np.mean(pset, axis=0)

    # Separate the center and the non-center points using specified algorithm
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

    if ret == 'cnc': # cnc = center + non-center
        return pscenter, prest
    elif ret == 'all':
        return pscenter, prest, pmean


def pointset_order(pset, center=None, direction='ccw'):
    """
    Order a point set around a center in a clockwise or counterclockwise way.

    :Parameters:
        pset : 2D array
            Pixel coordinates of the point set.
        center : list/tuple/1D array | None
            Pixel coordinates of the putative shape center.
        direction : str | 'ccw'
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


def pointset_locate(image, method='daofind', center='detected', centermethod='centroidnn',
                    direction='ccw', **kwds):
    """ A combination of detecting, sorting and ordering peaks from a 2D image.

    :Parameters:
        image : 2D array
            2D image for locating the point feature positions.
        method : str | 'daofind'
            Method for detecting peaks ('daofind' or 'maxlist').
        center : str/tuple/list
            Center position in (row, column) form.
        centermethod : str | 'centroidnn'
        direction : str | 'ccw'
            Direction of the ordering of the vertices ('cw' for clockwise, or 'ccw' for counterclockwise).
        **kwds : keyword arguments
            Extra arguments for the feature detection algorithms.
    """

    peaks = peakdetect2d(image, method=method, **kwds)
    if center == 'detected':
        pcenter, pverts = pointset_center(peaks, method=centermethod, ret='cnc')
        pverts_ord = pointset_order(pverts, center=None, direction=direction)

    elif type(center) in (list, tuple):
        pcenter = center
        pverts_ord = pointset_order(peaks, center=center, direction=direction)

    return pcenter, pverts_ord


def vvdist(verts, neighbor=1):
    """
    Calculate the neighboring vertex-vertex distance.

    :Parameters:
        verts : tuple/list
            Pixel coordinates of the vertices.
        neighbor : int | 1
            Neighbor index (1 = nearest).
    """

    if neighbor == 1:
        vvd = norm(verts - np.roll(verts, shift=-1, axis=0), axis=1)

    return vvd


def cvdist(verts, center):
    """
    Calculate the center-vertex distance.

    :Parameters:
        verts : tuple/list
            Pixel coordinates of the vertices.
        center : tuple/list
            Pixel coordinates of the center.
    """

    return norm(verts - center, axis=1)


def reorder(points, itemid, axis=0):
    """
    Reorder a point set along an axis.

    :Parameters:
        points : tuple/list
            Collection of the pixel coordinates of points.
        itemid : int
            Index of the entry to be placed at the start.
        axis : int | 1
            The axis to apply the shift.

    :Return:
        pts_rolled : tuple/list
            The points' pixel coordinates after position shift.
    """

    pts_rolled = np.roll(points, shift=itemid-1, axis=axis)

    return pts_rolled


def rotmat(theta, to_rad=True, coordsys='cartesian'):
    """ Rotation matrix in 2D in different coordinate systems.

    :Parameters:
        theta : numeric
            Rotation angle.
        to_rad : bool | True
            Specify the option to convert the angle to radians.
        coordsys : str | 'cartesian'
            Coordinate system specification ('cartesian' or 'homogen').
    """

    if to_rad:
        theta = np.radians(theta)

    c, s = np.cos(theta), np.sin(theta)
    if coordsys == 'cartesian':
        R = np.array([[c, -s], [s, c]])
    elif coordsys == 'homogen':
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    return R


def csm(pcent, pvert, rotsym=None, type='rotation'):
    """
    Computation of the continuous (a)symmetry measure (CSM) for a set of polygon
    vertices exhibiting a degree of rotational symmetry. The value is bounded within [0, 1].\n
    When csm = 0, the point set is completely symmetric.\n
    When csm = 1, the point set is completely asymmetric.

    :Parameters:
        pcent : tuple/list
            Pixel coordinates of the center position.
        pvert : numpy array
            Pixel coordinates of the vertices.
        rotsym : int | None
            Order of rotational symmetry.
        type : str | 'rotation'
            The type of the symmetry operation.

    Return:
        s : float
            Calculated continuous (a)symmetry measure.
    """

    if type == 'rotation':

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


def polyarea(x=[], y=[], coords=[], coord_order='rc'):
    """
    Calculate the area of a convex polygon area from its vertex coordinates, using
    the surveyor's formula (also called the shoelace formula).
    The vertices are ordered in a clockwise or counterclockwise fashions.

    :Parameters:
        x, y : tuple/list/1D array | [], []
            Collection of vertex coordinates along the x and y coordinates.
        coords : list/2D array | []
            Vertex coordinates.
        coord_order : str | 'rc'
            The ordering of coordinates in the `coords` array, choose from 'rc' or 'yx', 'cr' or 'xy'.
            Here r = row (y), c = column (x).

    :Return:
        A : numeric
            The area of the convex polygon bounded by the given vertices.
    """

    # If coords is specified, x and y arguments are ignored.
    if len(coords) > 0:
        if (coord_order == 'rc') or (coord_order == 'yx'):
            y, x = zip(*coords)
        elif (coord_order == 'cr') or (coord_order == 'xy'):
            x, y = zip(*coords)

    A = abs(sum(i * j for i, j in zip(x, y[1:] + y[:1]))
               - sum(i * j for i, j in zip(x[1:] + x[:1], y))) / 2

    return A


def arm(Aold, Anew):
    """
    Calculate the area retainment measure (ARM).

    :Parameters:
        Aold, Anew : numeric/numeric
            The area before (old) and after (new) symmetrization.

    :Return:
        s : numeric
            The value of the ARM.
    """

    rel = abs(1 - Anew/Aold)
    s = np.tanh(rel)

    return s


def gridplot(xgrid, ygrid, ax=None, subsamp=5, **kwds):
    """
    Plotting transform grid with downsampling. Adapted from the StackOverflow post,
    https://stackoverflow.com/questions/47295473/how-to-plot-using-matplotlib-python-colahs-deformed-grid

    :Parameters:
        xgrid, ygrid : 2D array, 2D array
            Coordinate grids along the x and y directions.
        ax : AxesObject
            Axes object to anchor the plot.
        subsamp : int | 5
            Subsampling portion.
        ``**kwds`` : keyword arguments
            Plotting keywords.
    """

    ny, nx = xgrid.shape

    # Subsampling the input grid coordinate matrices
    ss = int(subsamp)
    if ss != 1:
        xgrid = xgrid[::ss, ::ss]
        ygrid = ygrid[::ss, ::ss]
        nx, ny = nx // ss, ny // ss

    if ax is None:
        f, ax = plt.subplots(figsize=(4, 4))

    # Plot the y grid
    for i in range(ny):
        ax.plot(xgrid[i,:], ygrid[i,:], **kwds)

    # Plot the x grid
    for i in range(nx):
        ax.plot(xgrid[:,i], ygrid[:,i], **kwds)
