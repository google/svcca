# Copyright 2016 Google Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
The core code for applying Canonical Correlation Analysis to deep networks.

This module contains the core functions to apply canonical correlation analysis
to deep neural networks. The main function is get_cca_similarity, which takes in
two sets of activations, typically the neurons in two layers and their outputs
on all of the datapoints D = [d_1,...,d_m] that have been passed through.

Inputs have shape (num_neurons1, m), (num_neurons2, m). This can be directly
applied used on fully connected networks. For convolutional layers, the 3d block
of neurons can either be flattened entirely, along channels, or alternatively,
the dft_ccas (Discrete Fourier Transform) module can be used.

See https://arxiv.org/abs/1706.05806 for full details.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

num_cca_trials = 5
epsilon = 1e-6


def positivedef_matrix_sqrt(array):
  """Stable method for computing matrix square roots, supports complex matrices.

  Args:
            array: A numpy 2d array, can be complex valued that is a positive
                   definite symmetric (or hermitian) matrix

  Returns:
            sqrtarray: The matrix square root of array
  """
  w, v = np.linalg.eigh(array)
  #  A - np.dot(v, np.dot(np.diag(w), v.T))
  wsqrt = np.sqrt(w)
  sqrtarray = np.dot(v, np.dot(np.diag(wsqrt), np.conj(v).T))
  return sqrtarray


def remove_small(sigma_xx, sigma_xy, sigma_yx, sigma_yy, threshold=1e-6):
  """Takes covariance between X, Y, and removes values of small magnitude.

  Args:
            sigma_xx: 2d numpy array, variance matrix for x
            sigma_xy: 2d numpy array, crossvariance matrix for x,y
            sigma_yx: 2d numpy array, crossvariance matrixy for x,y,
                      (conjugate) transpose of sigma_xy
            sigma_yy: 2d numpy array, variance matrix for y
            threshold: cutoff value for norm below which directions are thrown
                       away

  Returns:
            sigma_xx_crop: 2d array with low x norm directions removed
            sigma_xy_crop: 2d array with low x and y norm directions removed
            sigma_yx_crop: 2d array with low x and y norm directiosn removed
            sigma_yy_crop: 2d array with low y norm directions removed
            x_idxs: indexes of sigma_xx that were removed
            y_idxs: indexes of sigma_yy that were removed
  """

  x_diag = np.abs(np.diagonal(sigma_xx))
  y_diag = np.abs(np.diagonal(sigma_yy))
  x_idxs = (x_diag >= threshold)
  y_idxs = (y_diag >= threshold)

  sigma_xx_crop = sigma_xx[x_idxs][:, x_idxs]
  sigma_xy_crop = sigma_xy[x_idxs][:, y_idxs]
  sigma_yx_crop = sigma_yx[y_idxs][:, x_idxs]
  sigma_yy_crop = sigma_yy[y_idxs][:, y_idxs]

  return (sigma_xx_crop, sigma_xy_crop, sigma_yx_crop, sigma_yy_crop, x_idxs,
          y_idxs)


def compute_ccas(sigma_xx, sigma_xy, sigma_yx, sigma_yy, verbose=True):
  """Main cca computation function, takes in variances and crossvariances.

  This function takes in the covariances and cross covariances of X, Y,
  preprocesses them (removing small magnitudes) and outputs the raw results of
  the cca computation, including cca directions in a rotated space, and the
  cca correlation coefficient values.

  Args:
            sigma_xx: 2d numpy array, (num_neurons_x, num_neurons_x)
                      variance matrix for x
            sigma_xy: 2d numpy array, (num_neurons_x, num_neurons_y)
                      crossvariance matrix for x,y
            sigma_yx: 2d numpy array, (num_neurons_y, num_neurons_x)
                      crossvariance matrix for x,y (conj) transpose of sigma_xy
            sigma_yy: 2d numpy array, (num_neurons_y, num_neurons_y)
                      variance matrix for y
            verbose:  boolean on whether to print intermediate outputs

  Returns:
            [ux, sx, vx]: [numpy 2d array, numpy 1d array, numpy 2d array]
                          ux and vx are (conj) transposes of each other, being
                          the canonical directions in the X subspace.
                          sx is the set of canonical correlation coefficients-
                          how well corresponding directions in vx, Vy correlate
                          with each other.
            [uy, sy, vy]: Same as above, but for Y space
            invsqrt_xx:   Inverse square root of sigma_xx to transform canonical
                          directions back to original space
            invsqrt_yy:   Same as above but for sigma_yy
            x_idxs:       The indexes of the input sigma_xx that were pruned
                          by remove_small
            y_idxs:       Same as above but for sigma_yy
  """

  (sigma_xx, sigma_xy, sigma_yx, sigma_yy, x_idxs, y_idxs) = remove_small(
      sigma_xx, sigma_xy, sigma_yx, sigma_yy)

  numx = sigma_xx.shape[0]
  numy = sigma_yy.shape[0]

  if numx == 0 or numy == 0:
    return ([0, 0, 0], [0, 0, 0], np.zeros_like(sigma_xx),
            np.zeros_like(sigma_yy), x_idxs, y_idxs)

  if verbose:
    print("adding eps to diagonal and taking inverse")
  sigma_xx +=epsilon * np.eye(numx)
  sigma_yy +=epsilon * np.eye(numy)
  inv_xx = np.linalg.pinv(sigma_xx)
  inv_yy = np.linalg.pinv(sigma_yy)

  if verbose:
    print("taking square root")
  invsqrt_xx = positivedef_matrix_sqrt(inv_xx)
  invsqrt_yy = positivedef_matrix_sqrt(inv_yy)

  if verbose:
    print("dot products...")
  arr_x = np.dot(sigma_yx, invsqrt_xx)
  arr_x = np.dot(inv_yy, arr_x)
  arr_x = np.dot(invsqrt_xx, np.dot(sigma_xy, arr_x))
  arr_y = np.dot(sigma_xy, invsqrt_yy)
  arr_y = np.dot(inv_xx, arr_y)
  arr_y = np.dot(invsqrt_yy, np.dot(sigma_yx, arr_y))

  if verbose:
    print("trying to take final svd")
  arr_x_stable = arr_x + epsilon * np.eye(arr_x.shape[0])
  arr_y_stable = arr_y + epsilon * np.eye(arr_y.shape[0])
  ux, sx, vx = np.linalg.svd(arr_x_stable)
  uy, sy, vy = np.linalg.svd(arr_y_stable)

  sx = np.sqrt(np.abs(sx))
  sy = np.sqrt(np.abs(sy))
  if verbose:
    print("computed everything!")

  return [ux, sx, vx], [uy, sy, vy], invsqrt_xx, invsqrt_yy, x_idxs, y_idxs


def sum_threshold(array, threshold):
  """Computes threshold index of decreasing nonnegative array by summing.

  This function takes in a decreasing array nonnegative floats, and a
  threshold between 0 and 1. It returns the index i at which the sum of the
  array up to i is threshold*total mass of the array.

  Args:
            array: a 1d numpy array of decreasing, nonnegative floats
            threshold: a number between 0 and 1

  Returns:
            i: index at which np.sum(array[:i]) >= threshold
  """
  assert (threshold >= 0) and (threshold <= 1), "print incorrect threshold"

  for i in xrange(len(array)):
    if np.sum(array[:i]) / np.sum(array) >= threshold:
      return i


def create_zero_dict(compute_dirns, dimension):
  """Outputs a zero dict when neuron activation norms too small.

  This function creates a return_dict with appropriately shaped zero entries
  when all neuron activations are very small.

  Args:
            compute_dirns: boolean, whether to have zero vectors for directions
            dimension: int, defines shape of directions

  Returns:
            return_dict: a dict of appropriately shaped zero entries
  """
  return_dict = {}
  return_dict["mean"] = (np.asarray(0), np.asarray(0))
  return_dict["sum"] = (np.asarray(0), np.asarray(0))
  return_dict["cca_coef1"] = np.asarray(0)
  return_dict["cca_coef2"] = np.asarray(0)
  return_dict["idx1"] = 0
  return_dict["idx2"] = 0

  if compute_dirns:
    return_dict["cca_dirns1"] = np.zeros((1, dimension))
    return_dict["cca_dirns2"] = np.zeros((1, dimension))

  return return_dict


def get_cca_similarity(acts1, acts2, threshold=0.98, compute_dirns=True,
                       verbose=True):
  """The main function for computing cca similarities.

  This function computes the cca similarity between two sets of activations,
  returning a dict with the cca coefficients, a few statistics of the cca
  coefficients, and (optionally) the actual directions.

  Args:
            acts1: (num_neurons1, data_points) a 2d numpy array of neurons by
                   datapoints where entry (i,j) is the output of neuron i on
                   datapoint j.
            acts2: (num_neurons2, data_points) same as above, but (potentially)
                   for a different set of neurons. Note that acts1 and acts2
                   can have different numbers of neurons, but must agree on the
                   number of datapoints

            threshold: float between 0, 1 used to get rid of trailing zeros in
                       the cca correlation coefficients to output more accurate
                       summary statistics of correlations.

            compute_dirns: boolean value determining whether actual cca
                           directions are computed. (For very large neurons and
                           datasets, may be better to compute these on the fly
                           instead of store in memory.)

            verbose: Boolean, whether info about intermediate outputs printed

  Returns:
            return_dict: A dictionary with outputs from the cca computations.
                         Contains neuron coefficients (combinations of neurons
                         that correspond to cca directions), the cca correlation
                         coefficients (how well aligned directions correlate),
                         x and y idxs (for computing cca directions on the fly
                         if compute_dirns=False), and summary statistics. If
                         compute_dirns=True, the cca directions are also
                         computed.
  """

  # assert dimensionality equal
  assert acts1.shape[1] == acts2.shape[1], "dimensions don't match"
  # check that acts1, acts2 are transposition
  assert acts1.shape[0] < acts1.shape[1], ("input must be number of neurons"
                                           "by datapoints")
  return_dict = {}

  # compute covariance with numpy function for extra stability
  numx = acts1.shape[0]

  covariance = np.cov(acts1, acts2)
  sigmaxx = covariance[:numx, :numx]
  sigmaxy = covariance[:numx, numx:]
  sigmayx = covariance[numx:, :numx]
  sigmayy = covariance[numx:, numx:]

  # rescale covariance to make cca computation more stable
  xmax = np.max(np.abs(sigmaxx))
  ymax = np.max(np.abs(sigmayy))
  sigmaxx /= xmax
  sigmayy /= ymax
  sigmaxy /= np.sqrt(xmax * ymax)
  sigmayx /= np.sqrt(xmax * ymax)

  ([_, sx, vx], [_, sy, vy], invsqrt_xx, invsqrt_yy, x_idxs,
   y_idxs) = compute_ccas(sigmaxx, sigmaxy, sigmayx, sigmayy,
                          verbose)

  # if x_idxs or y_idxs is all false, return_dict has zero entries
  if (not np.any(x_idxs)) or (not np.any(y_idxs)):
    return create_zero_dict(compute_dirns, acts1.shape[1])

  if compute_dirns:
    # orthonormal directions that are CCA directions
    cca_dirns1 = np.dot(vx, np.dot(invsqrt_xx, acts1[x_idxs]))
    cca_dirns2 = np.dot(vy, np.dot(invsqrt_yy, acts2[y_idxs]))

  # get rid of trailing zeros in the cca coefficients
  idx1 = sum_threshold(sx, threshold)
  idx2 = sum_threshold(sy, threshold)

  return_dict["neuron_coeffs1"] = np.dot(vx, invsqrt_xx)
  return_dict["neuron_coeffs2"] = np.dot(vy, invsqrt_yy)
  return_dict["cca_coef1"] = sx
  return_dict["cca_coef2"] = sy
  return_dict["x_idxs"] = x_idxs
  return_dict["y_idxs"] = y_idxs
  # summary statistics
  return_dict["mean"] = (np.mean(sx[:idx1]), np.mean(sy[:idx2]))
  return_dict["sum"] = (np.sum(sx), np.sum(sy))

  if compute_dirns:
    return_dict["cca_dirns1"] = cca_dirns1
    return_dict["cca_dirns2"] = cca_dirns2

  return return_dict


def robust_cca_similarity(acts1, acts2, threshold=0.98, compute_dirns=True):
  """Calls get_cca_similarity multiple times while adding noise.

  This function is very similar to get_cca_similarity, and can be used if
  get_cca_similarity doesn't converge for some pair of inputs. This function
  adds some noise to the activations to help convergence.

  Args:
            acts1: (num_neurons1, data_points) a 2d numpy array of neurons by
                   datapoints where entry (i,j) is the output of neuron i on
                   datapoint j.
            acts2: (num_neurons2, data_points) same as above, but (potentially)
                   for a different set of neurons. Note that acts1 and acts2
                   can have different numbers of neurons, but must agree on the
                   number of datapoints

            threshold: float between 0, 1 used to get rid of trailing zeros in
                       the cca correlation coefficients to output more accurate
                       summary statistics of correlations.

            compute_dirns: boolean value determining whether actual cca
                           directions are computed. (For very large neurons and
                           datasets, may be better to compute these on the fly
                           instead of store in memory.)

  Returns:
            return_dict: A dictionary with outputs from the cca computations.
                         Contains neuron coefficients (combinations of neurons
                         that correspond to cca directions), the cca correlation
                         coefficients (how well aligned directions correlate),
                         x and y idxs (for computing cca directions on the fly
                         if compute_dirns=False), and summary statistics. If
                         compute_dirns=True, the cca directions are also
                         computed.
  """

  for trial in xrange(num_cca_trials):
    try:
      return_dict = get_cca_similarity(acts1, acts2, threshold, compute_dirns)
    except np.LinAlgError:
      acts1 = acts1 * 1e-1 + np.random.normal(size=acts1.shape) * epsilon
      acts2 = acts2 * 1e-1 + np.random.normal(size=acts1.shape) * epsilon
      if trial + 1 == num_cca_trials:
        raise

  return return_dict
