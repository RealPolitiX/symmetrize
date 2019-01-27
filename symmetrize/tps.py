#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.ndimage as ndi


# Thin-plate spline adapted from Zachary Pincus' implementation in celltool
# https://github.com/zpincus/celltool

def tpsWarping(from_points, to_points, images, axis=2, interpolation_order=1, approximate_grid=1, **kwds):
    """
    Calculate the thin-plate spline (TPS) warping transform that from the from_points
    to the to_points, and then warp the given images by that transform. This
    transform is described in the paper: "Principal Warps: Thin-Plate Splines and
    the Decomposition of Deformations" by F.L. Bookstein.

    :Parameters:
        from_points, to_points : 2D array, 2D array (dim = n x 2)
            Correspondence point sets containing n 2D landmarks from the distorted and ideal images.
            The coordinates are in the (row, column) convention.
        images : 3D array
            3D image to warp with the calculated thin-plate spline transform.
        axis : int | 2
            Axis to perform the warping operation.
        interpolation_order : int | 1
            If 1, then use linear interpolation; if 0 then use nearest-neighbor.
            See `scipy.ndimage.map_coordinates()`.
        approximate_grid : int | 1
            Use the approximate grid (if set > 1) for the transform. The approximate grid is smaller
            than the output image region, and then the transform is bilinearly interpolated to the
            larger region. This is fairly accurate for values up to 10 or so.
        **kwds : keyword arguments
            :output_region: tuple | (0, 0, # of columns in image, # of rows in image)
                The (xmin, ymin, xmax, ymax) region of the output image that should be produced.
                (Note: The region is inclusive, i.e. xmin <= x <= xmax).
            :ret: str | 'all'
                Function return specification.
                `'image'`: return the transformed image.
                `'deform'` : return the deformation field.
                `'all'`: return both the transformed images and deformation field.

    :Returns:
        images_tf : nD array
            Transformed image stack.
        transform : list
            Deformation field along x and y axes.
    """

    images = np.moveaxis(images, axis, 0)
    nim, nr, nc = images.shape
    output_region = kwds.pop('output_region', (0, 0, nc, nr))
    ret = kwds.pop('ret', 'all')

    transform = _make_inverse_warp(from_points, to_points, output_region, approximate_grid)
    images_tf = np.asarray([ndi.map_coordinates(image, transform, order=interpolation_order) for image in list(images)])
    images_tf = np.moveaxis(images_tf, 0, axis)

    if ret == 'all':
        return images_tf, transform
    elif ret == 'image':
        return images_tf
    elif ret == 'deform':
        return transform


def _make_inverse_warp(from_points, to_points, output_region, approximate_grid):
    """
    Calculate the warping transform.
    """

    x_min, y_min, x_max, y_max = output_region

    if approximate_grid is None:
        approximate_grid = 1

    x_steps = (x_max - x_min) / approximate_grid
    y_steps = (y_max - y_min) / approximate_grid
    x, y = np.mgrid[x_min:x_max:x_steps*1j, y_min:y_max:y_steps*1j]

    # make the reverse transform warping from the to_points to the from_points, because we
    # do image interpolation in this reverse fashion
    transform = _make_warp(to_points, from_points, x, y)

    if approximate_grid != 1:

        # linearly interpolate the zoomed transform grid
        new_x, new_y = np.mgrid[x_min:x_max+1, y_min:y_max+1]
        x_fracs, x_indices = np.modf((x_steps-1)*(new_x-x_min)/float(x_max-x_min))
        y_fracs, y_indices = np.modf((y_steps-1)*(new_y-y_min)/float(y_max-y_min))
        x_indices = x_indices.astype(int)
        y_indices = y_indices.astype(int)
        x1 = 1 - x_fracs
        y1 = 1 - y_fracs
        ix1 = (x_indices+1).clip(0, x_steps-1)
        iy1 = (y_indices+1).clip(0, y_steps-1)

        t00 = transform[0][(x_indices, y_indices)]
        t01 = transform[0][(x_indices, iy1)]
        t10 = transform[0][(ix1, y_indices)]
        t11 = transform[0][(ix1, iy1)]
        transform_x = t00*x1*y1 + t01*x1*y_fracs + t10*x_fracs*y1 + t11*x_fracs*y_fracs

        t00 = transform[1][(x_indices, y_indices)]
        t01 = transform[1][(x_indices, iy1)]
        t10 = transform[1][(ix1, y_indices)]
        t11 = transform[1][(ix1, iy1)]
        transform_y = t00*x1*y1 + t01*x1*y_fracs + t10*x_fracs*y1 + t11*x_fracs*y_fracs

        transform = [transform_x, transform_y]

    return transform

_small = 1e-10


def _U(x):

    return (x**2) * np.where(x<_small, 0, np.log(x))


def _interpoint_distances(points):
    """
    Calculate the pair distance within a point set.
    """

    xd = np.subtract.outer(points[:,0], points[:,0])
    yd = np.subtract.outer(points[:,1], points[:,1])

    return np.sqrt(xd**2 + yd**2)


def _make_L_matrix(points):
    """
    Construct the L matrix following Bookstein's description.
    """

    n = len(points)
    K = _U(_interpoint_distances(points))
    P = np.ones((n, 3))
    P[:,1:] = points
    O = np.zeros((3, 3))
    # Construct L matrix from constituent blocks
    L = np.asarray(np.bmat([[K, P], [P.transpose(), O]]))

    return L


def _calculate_f(coeffs, points, x, y):
    """
    Calculate the thin plate energy function.
    """

    w = coeffs[:-3]
    a1, ax, ay = coeffs[-3:]
    # The following uses too much RAM:
    # distances = _U(numpy.sqrt((points[:,0]-x[...,numpy.newaxis])**2 + (points[:,1]-y[...,numpy.newaxis])**2))
    # summation = (w * distances).sum(axis=-1)
    summation = np.zeros(x.shape)

    for wi, Pi in zip(w, points):
        summation += wi * _U(np.sqrt((x-Pi[0])**2 + (y-Pi[1])**2))

    return a1 + ax*x + ay*y + summation


def _make_warp(from_points, to_points, x_vals, y_vals):
    """
    Calculate the pixel warping displacement for the x and y coordinates.
    """

    from_points, to_points = np.asarray(from_points), np.asarray(to_points)
    err = np.seterr(divide='ignore')
    L = _make_L_matrix(from_points)

    V = np.resize(to_points, (len(to_points)+3, 2))
    V[-3:, :] = 0
    coeffs = np.dot(np.linalg.pinv(L), V)

    x_warp = _calculate_f(coeffs[:,0], from_points, x_vals, y_vals)
    y_warp = _calculate_f(coeffs[:,1], from_points, x_vals, y_vals)
    np.seterr(**err)

    return [x_warp, y_warp]
